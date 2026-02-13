
# app/ai/optimizer.py
"""
Handles post-export optimization of ONNX models, specifically quantization to INT8.
"""

import contextlib
import logging
import os
import shutil
from pathlib import Path

from app.shared.constants import APP_TEMP_DIR, QuantizationMode

logger = logging.getLogger("PixelHand.ai.optimizer")


def _sanitize_model_shapes(input_path: Path, output_path: Path):
    """
    Loads an ONNX model, strips explicit shape information from ValueInfo and Outputs,
    and saves it to a new path. This fixes 'ShapeInferenceError' where the exported
    model contains fixed shapes that conflict with ONNX Runtime's inference logic.
    """
    try:
        import onnx

        # CRITICAL FIX: Load with external data (True) to pull weights into memory.
        # If we use False, onnx.save writes relative paths pointing to the original
        # data file location, which is invalid when saving to the separate 'temp' dir.
        model = onnx.load(str(input_path), load_external_data=True)

        # 1. Clear inferred value info (intermediate shapes)
        del model.graph.value_info[:]

        # 2. Clear output shapes (let inference deduce them)
        for output in model.graph.output:
            if output.type.tensor_type.HasField("shape"):
                output.type.tensor_type.ClearField("shape")

        # 3. Save sanitized copy to temp dir.
        # explicitly save external data to a new file in the temp dir so the
        # sanitized model is self-contained and valid for quantization tools.
        onnx.save(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.name}.data",
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to sanitize model shapes: {e}")
        return False


def optimize_model_post_export(
    visual_path: Path,
    text_path: Path,
    quant_mode: QuantizationMode,
):
    """
    Performs post-export optimization (INT8 Dynamic Quantization) on the given model files.
    """
    if quant_mode != QuantizationMode.INT8:
        return

    # Store the original working directory
    original_cwd = os.getcwd()

    try:
        from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

        logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        # Force the process to work inside the temp directory for intermediate files.
        # Note: We pass ABSOLUTE paths for inputs to ensure external data is found.
        os.chdir(APP_TEMP_DIR)

        def _process_single_model(final_path: Path):
            if not final_path.exists():
                return

            # Intermediate paths in temp directory
            # We use a unique name to avoid conflicts if multiple runs happen
            temp_preproc = APP_TEMP_DIR / f"{final_path.stem}_preproc.onnx"
            temp_sanitized = APP_TEMP_DIR / f"{final_path.stem}_sanitized.onnx"
            temp_quant = APP_TEMP_DIR / f"{final_path.stem}_quant.onnx"

            # Start with original file as input
            current_input_model = str(final_path)
            pre_process_success = False

            # 1. Try Pre-processing (Shape Inference)
            # This handles large models. We save external data to temp dir if needed.
            # CRITICAL: We read from 'final_path' in its original location so it finds its .data files.
            try:
                quant_pre_process(
                    input_model_path=str(final_path),
                    output_model_path=str(temp_preproc),
                    skip_optimization=False,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )
                current_input_model = str(temp_preproc)
                pre_process_success = True
            except Exception as e:
                # Common error: "list index out of range" in older ORT versions or specific models
                logger.warning(f"INT8 Pre-processing skipped for {final_path.name} (using raw FP32 fallback). Reason: {e}")

            # 2. Run Dynamic Quantization
            try:
                # Attempt standard quantization
                quantize_dynamic(
                    model_input=current_input_model,
                    model_output=str(temp_quant),
                    weight_type=QuantType.QUInt8,
                    extra_options={'DisableShapeInference': True} # Try to disable, though ORT often ignores this in init
                )
            except Exception as e:
                # If it failed (likely ShapeInferenceError), and we haven't pre-processed successfully
                # (because pre-processing usually fixes shapes), try sanitizing the raw model.
                if not pre_process_success:
                    logger.warning(f"Standard quantization failed ({e}). Retrying with shape sanitization...")

                    if _sanitize_model_shapes(final_path, temp_sanitized):
                        try:
                            quantize_dynamic(
                                model_input=str(temp_sanitized),
                                model_output=str(temp_quant),
                                weight_type=QuantType.QUInt8,
                                extra_options={'DisableShapeInference': True}
                            )
                            logger.info(f"Quantization successful after sanitization for {final_path.name}")
                        except Exception as e2:
                            logger.error(f"Retry failed for {final_path.name}: {e2}")
                            raise e2
                    else:
                        raise e
                else:
                    raise e

            # 3. Success: Overwrite the original file with the quantized version

            # First, remove original external data if it exists (cleanup old FP32 data)
            # Common patterns: .data, .onnx.data, _data
            # We do this before moving to ensure a clean state
            for garbage in final_path.parent.glob(f"{final_path.name}*data"):
                with contextlib.suppress(OSError):
                    garbage.unlink()

            # Move quantized file to final location, overwriting the FP32 onnx
            shutil.move(str(temp_quant), str(final_path))

            # Cleanup Temps
            with contextlib.suppress(OSError):
                if temp_preproc.exists():
                    temp_preproc.unlink()
                if temp_sanitized.exists():
                    temp_sanitized.unlink()
                if temp_quant.exists():
                    temp_quant.unlink() # In case move failed or wasn't reached

                # Cleanup external data generated by pre-process in temp dir
                for f in APP_TEMP_DIR.glob(f"{final_path.stem}_preproc.onnx.*"):
                    f.unlink()
                for f in APP_TEMP_DIR.glob(f"{final_path.stem}_sanitized.onnx.*"):
                    f.unlink()

        logger.warning("Starting INT8 Quantization (this may take a moment)...")

        # Resolve paths to ensure they are absolute
        _process_single_model(visual_path.resolve())
        _process_single_model(text_path.resolve())

        logger.warning("INT8 Quantization complete.")
        root_logger.setLevel(original_level)

    except ImportError:
        logger.error("onnxruntime not installed, cannot perform quantization.")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        if "root_logger" in locals():
            root_logger.setLevel(original_level)
        raise e
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)
