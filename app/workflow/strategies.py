# app/workflow/strategies.py
"""
Scanning Strategies.
Implements the Template Method pattern to standardize scan execution flow.
"""

import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

# Infrastructure & Domain
from app.domain.config import ScanConfig
from app.domain.data_models import (
    AnalysisItem,
    DuplicateResults,
    EvidenceMethod,
    ImageFingerprint,
    ScanMode,
    ScanState,
)

# Workflow Components
from app.imaging.image_io import get_image_metadata, load_image
from app.infrastructure.container import ServiceContainer
from app.shared.constants import (
    BEST_FILE_METHOD_NAME,
    LANCEDB_AVAILABLE,
    SIMILARITY_SEARCH_K_NEIGHBORS,
)
from app.shared.signal_bus import SignalBus
from app.workflow.auxiliary import FileFinder
from app.workflow.pipeline import PipelineManager
from app.workflow.qc_rules import QCRules
from app.workflow.stages import (
    AILinkingStage,
    DatabaseIndexStage,
    ExactDuplicateStage,
    FileDiscoveryStage,
    FingerprintGenerationStage,
    ItemGenerationStage,
    PerceptualDuplicateStage,
    ScanContext,
    ScanStage,
)

logger = logging.getLogger("PixelHand.strategies")


class ScanStrategy(ABC):
    """
    Abstract base class for a scanning strategy.
    Implements the Template Method 'execute' to handle boilerplate.
    """

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: SignalBus,
        scanner_core: Any,
        services: ServiceContainer,
    ):
        self.config = config
        self.state = state
        self.signals = signals
        self.scanner_core = scanner_core
        self.services = services
        self.all_skipped_files: list[str] = []

    def execute(self, stop_event: threading.Event, start_time: float):
        """
        The Template Method. Orchestrates the common lifecycle of a scan.
        """
        # 1. Common File Finding (Hook)
        # Some strategies might skip this or do it differently, but generally we need files.
        try:
            files = self._find_files_as_list(stop_event)
        except Exception as e:
            logger.error(f"File finding failed: {e}")
            self.scanner_core._finalize_scan(None, 0, self.config.scan_mode, 0, [])
            return

        # 2. Check early exit conditions
        if self._check_early_stop(stop_event, files, start_time):
            return

        # 3. Strategy Specific Logic (Abstract Hook)
        # Should return a tuple: (final_payload, num_found, mode)
        # or None if it handled finalization itself.
        result_pkg = self._run_strategy_logic(stop_event, start_time, files)

        # 4. Finalize
        if result_pkg:
            payload, num_found, mode = result_pkg
            duration = time.time() - start_time
            self.scanner_core._finalize_scan(
                payload, num_found, mode, duration, self.all_skipped_files
            )

    @abstractmethod
    def _run_strategy_logic(
        self, stop_event: threading.Event, start_time: float, files: list[Path]
    ) -> tuple[dict | None, int, ScanMode] | None:
        """
        Specific logic for the strategy.
        Returns: (payload, num_found, scan_mode) OR None if already finalized.
        """
        pass

    def _find_files_as_list(self, stop_event: threading.Event) -> list[Path]:
        """Helper to scan the filesystem using FileFinder."""
        self.state.set_phase("Finding image files...", 0.1)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        # Consume the generator
        files = [
            path
            for batch in finder.stream_files(stop_event)
            if not stop_event.is_set()
            for path, _ in batch
        ]
        files.sort()
        return files

    def _check_early_stop(self, stop_event, collection, start_time) -> bool:
        """Checks cancellation or empty results."""
        return self.scanner_core._check_stop_or_empty(
            stop_event,
            collection,
            self.config.scan_mode,
            {"db_path": None},
            start_time,
        )

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Creates a placeholder fingerprint for metadata-only items."""
        try:
            meta = get_image_metadata(path)
            if meta:
                return ImageFingerprint(path=path, hashes=np.array([]), **meta)
        except Exception:
            self.all_skipped_files.append(str(path))
        return None


class FindDuplicatesStrategy(ScanStrategy):
    """
    Strategy for finding duplicate images using a granular pipeline of stages.
    """

    def _run_strategy_logic(self, stop_event, start_time, files):
        # We ignore 'files' arg because FileDiscoveryStage handles it in the pipeline
        # but the base class template fetches it. We can repurpose it or let the pipeline re-do it.
        # Optimization: Pass the found files to context to avoid re-scanning.

        context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            lancedb_db=self.services.db_service.db,
            lancedb_table=self.services.db_service.table,
            db_service=self.services.db_service,
            files_to_process=files,  # Pre-filled from template
        )

        pipeline_to_run = self._build_pipeline()

        for stage, weight in pipeline_to_run:
            if stop_event.is_set():
                break
            if isinstance(stage, FileDiscoveryStage) and files:
                continue  # Skip discovery if template already did it

            self.state.set_phase(stage.name, weight)
            if not stage.run(context):
                break

        self.all_skipped_files.extend(context.all_skipped_files)

        if stop_event.is_set():
            return None  # Will be handled by cancellation logic in caller or manual finalize?
            # Actually template handles finalize if we return.
            # But duplicate strategy has specific finalize logic.

        return self._prepare_results_package(context)

    def _build_pipeline(self) -> list[tuple[ScanStage, float]]:
        pipeline = []
        # If we pre-filled files, we don't need FileDiscoveryStage strictly,
        # but keeping it for architecture consistency if we didn't pass files.
        # Here we assume we use the pre-filled files.

        if self.config.hashing.find_exact:
            pipeline.append((ExactDuplicateStage(), 0.10))

        needs_items = (
            self.config.hashing.find_simple
            or self.config.hashing.find_perceptual
            or self.config.hashing.find_structural
            or self.config.ai.use_ai
        )
        if needs_items:
            pipeline.append((ItemGenerationStage(), 0.05))

        if any(
            [
                self.config.hashing.find_simple,
                self.config.hashing.find_perceptual,
                self.config.hashing.find_structural,
            ]
        ):
            pipeline.append((PerceptualDuplicateStage(), 0.20))

        if self.config.ai.use_ai:
            pipeline.extend(
                [
                    (FingerprintGenerationStage(), 0.40),
                    (DatabaseIndexStage(), 0.05),
                    (AILinkingStage(), 0.15),
                ]
            )
        else:
            self.signals.log_message.emit("Running in 'No AI' mode.", "info")

        return pipeline

    def _prepare_results_package(self, context: ScanContext):
        self.state.set_phase("Optimizing lookup tables...", 0.01)
        str_map = {str(k): v for k, v in context.all_image_fps.items()}

        def fp_resolver(node: tuple[str, str | None]) -> ImageFingerprint | None:
            path_str, channel = node
            if fp_orig := str_map.get(path_str):
                fp_copy = copy.copy(fp_orig)
                fp_copy.channel = channel
                return fp_copy
            return None

        self.state.set_phase("Finalizing groups...", 0.05)
        final_groups = context.cluster_manager.get_final_groups(fp_resolver)

        num_found = sum(len(d) for d in final_groups.values())
        db_uri = str(self.services.db_service.db.uri) if self.services.db_service.db else None
        groups_summary = []

        if num_found > 0 and LANCEDB_AVAILABLE:
            try:
                self.state.set_phase("Saving results to database...", 0.0)
                self._persist_results_to_lancedb(final_groups)
                group_id = 1
                for best_fp, dups in final_groups.items():
                    total_size = best_fp.file_size + sum(fp.file_size for fp, _, _ in dups)
                    group_name = best_fp.path.stem
                    if best_fp.channel:
                        group_name += f" ({best_fp.channel})"
                    groups_summary.append(
                        {
                            "group_id": group_id,
                            "name": group_name,
                            "count": len(dups) + 1,
                            "total_size": total_size,
                        }
                    )
                    group_id += 1
            except Exception as e:
                logger.error(f"Failed to persist results: {e}", exc_info=True)
                self.signals.scan_error.emit(f"Result persistence failed: {e}")
                return None

        payload = {
            "db_path": db_uri,
            "groups_data": None,
            "lazy_summary": groups_summary,
        }
        return payload, num_found, ScanMode.DUPLICATES

    def _persist_results_to_lancedb(self, final_groups: DuplicateResults):
        data = []
        group_id = 1
        for best_fp, dups in final_groups.items():
            row = best_fp.to_lancedb_dict()
            row.update(
                {
                    "group_id": group_id,
                    "is_best": True,
                    "distance": 0,
                    "found_by": BEST_FILE_METHOD_NAME,
                }
            )
            self._clean_row_for_results_table(row)
            data.append(row)
            for dup_fp, score, method in dups:
                row = dup_fp.to_lancedb_dict()
                row.update(
                    {
                        "group_id": group_id,
                        "is_best": False,
                        "distance": score,
                        "found_by": method,
                    }
                )
                self._clean_row_for_results_table(row)
                data.append(row)
            group_id += 1

        if data:
            self.services.db_service.save_results_table(data)

    def _clean_row_for_results_table(self, row: dict):
        row.pop("vector", None)
        row.pop("id", None)


class SearchStrategy(ScanStrategy):
    """
    Strategy for Text and Sample Image search.
    """

    def _run_strategy_logic(self, stop_event, start_time, all_files):
        # Template already found files
        self.state.set_phase("Phase 2/2: Creating AI fingerprints...", 0.8)

        pipeline_manager = PipelineManager(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            services=self.services,
        )

        items = [AnalysisItem(path=path, analysis_type="Composite") for path in all_files]
        fps_map = {}
        for item in items:
            if item.path not in fps_map:
                meta = get_image_metadata(item.path)
                if meta:
                    fps_map[item.path] = ImageFingerprint(path=item.path, hashes=np.array([]), **meta)
                else:
                    self.all_skipped_files.append(str(item.path))

        search_context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            lancedb_db=self.services.db_service.db,
            lancedb_table=self.services.db_service.table,
            db_service=self.services.db_service,
            items_to_process=items,
            all_image_fps=fps_map,
        )

        success, skipped = pipeline_manager.run(search_context)
        self.all_skipped_files.extend(skipped)

        if not success and not stop_event.is_set():
            self.signals.scan_error.emit("Failed to generate fingerprints.")
            return None

        num_found, db_path, results_list = self._perform_similarity_search()
        if num_found is None:
            return None

        final_groups = self._create_search_groups(results_list)
        if num_found > 0:
            self._persist_search_results(final_groups)

        return (
            {"db_path": db_path, "groups_data": final_groups, "lazy_summary": None},
            num_found,
            self.config.scan_mode,
        )

    def _perform_similarity_search(self):
        self.state.set_phase("Searching for similar images...", 0.1)
        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.scan_error.emit("Could not generate a vector for the search query.")
            return None, None, []

        threshold = getattr(self.config, "similarity_threshold", 70)
        dist_threshold = 1.0 - (threshold / 100.0)
        results = []

        if self.services.db_service.is_ready:
            try:
                precision_key = self.config.perf.search_precision
                from app.shared.constants import (
                    DEFAULT_SEARCH_PRECISION,
                    SEARCH_PRECISION_PRESETS,
                )

                precision_config = SEARCH_PRECISION_PRESETS.get(
                    precision_key, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
                )

                raw_rows = self.services.db_service.search_vectors(
                    query_vector,
                    limit=SIMILARITY_SEARCH_K_NEIGHBORS,
                    probes=precision_config["nprobes"],
                    refine=precision_config["refine_factor"],
                )

                for row_dict in raw_rows:
                    dist = row_dict["_distance"]
                    if dist < dist_threshold:
                        results.append((ImageFingerprint.from_db_row(row_dict), dist))
            except Exception as e:
                self.signals.log_message.emit(f"LanceDB search failed: {e}", "error")
                return None, None, []
        else:
            self.signals.scan_error.emit("Database service not ready.")
            return None, None, []

        return len(results), str(self.services.db_service.db.uri), results

    def _get_query_vector(self) -> np.ndarray | None:
        try:
            ai_config_dict = self.config.ai.__dict__
            self.services.model_manager.ensure_model_loaded(ai_config_dict)
            engine = self.services.model_manager.current_engine
            if not engine:
                return None

            if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
                return engine.get_text_features(self.config.search_query)

            elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
                from app.ai.preprocessing import ImageBatchPreprocessor

                items = [AnalysisItem(path=self.config.sample_path, analysis_type="Composite")]
                images, _, _ = ImageBatchPreprocessor.prepare_batch(items, engine.input_size)

                if images:
                    px = engine.processor(images=images, return_tensors="np").pixel_values
                    if engine.is_fp16:
                        px = px.astype(np.float16)
                    io = engine.visual_session.io_binding()
                    io.bind_cpu_input("pixel_values", px)
                    io.bind_output("image_embeds")
                    engine.visual_session.run_with_iobinding(io)
                    from app.ai.manager import normalize_vectors_numpy

                    return normalize_vectors_numpy(io.get_outputs()[0].numpy()).flatten()
        except Exception as e:
            logger.error(f"Query vector generation failed: {e}")
        return None

    def _create_search_groups(self, results) -> DuplicateResults:
        best_fp = self._create_search_context_fingerprint()
        if not best_fp:
            return {}
        dups = set()
        for fp, d in results:
            score = int(max(0.0, (1.0 - d) * 100))
            dups.add((fp, score, EvidenceMethod.AI.value))
        return {best_fp: dups}

    def _persist_search_results(self, final_groups):
        data = []
        group_id = 1
        for best_fp, dups in final_groups.items():
            row = best_fp.to_lancedb_dict()
            row.update({"group_id": group_id, "is_best": True, "distance": 0, "found_by": "Query"})
            self._clean_row(row)
            data.append(row)
            for dup_fp, score, method in dups:
                row = dup_fp.to_lancedb_dict()
                row.update(
                    {"group_id": group_id, "is_best": False, "distance": score, "found_by": method}
                )
                self._clean_row(row)
                data.append(row)
            group_id += 1
        if data:
            self.services.db_service.save_results_table(data)

    def _clean_row(self, row):
        if "vector" in row:
            del row["vector"]
        if "id" in row:
            del row["id"]

    def _create_search_context_fingerprint(self) -> ImageFingerprint | None:
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return self._create_dummy_fp(self.config.sample_path)
        return ImageFingerprint(
            path=Path(f"Query: '{self.config.search_query}'"),
            hashes=np.array([]),
            resolution=(0, 0),
            file_size=0,
            mtime=0,
            capture_date=None,
            format_str="SEARCH",
            compression_format="Text Query",
            format_details="Text Query",
            has_alpha=False,
            bit_depth=8,
            mipmap_count=1,
            texture_type="2D",
            color_space="N/A",
        )


class FolderComparisonStrategy(ScanStrategy):
    """
    Strategy for comparing two folders.
    """

    def _run_strategy_logic(self, stop_event, start_time, files_a_list):
        folder_b = self.config.comparison_folder_path

        if not folder_b:
            self.signals.scan_error.emit("Secondary folder not specified.")
            return None

        def get_key(p: Path):
            return p.stem.lower() if self.config.qc.match_by_stem else p.name.lower()

        # Files from Folder A are passed in via template (files_a_list)
        files_a = {get_key(p): p for p in files_a_list}

        # Scan Folder B
        self.state.set_phase(f"Scanning Folder B: {folder_b.name}...", 0.1)
        finder_b = FileFinder(
            self.state,
            folder_b,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files_b = {
            get_key(p): p
            for batch in finder_b.stream_files(stop_event)
            for p, _ in batch
        }
        if stop_event.is_set():
            return None

        common_keys = set(files_a.keys()) & set(files_b.keys())
        self.state.set_phase("Comparing metadata & QC...", 0.8)

        final_groups: DuplicateResults = {}
        processed = 0
        needs_image = self.config.qc.check_solid_color or self.config.qc.check_normal_maps
        from PIL import ImageStat

        for key in common_keys:
            if stop_event.is_set():
                break

            path_a, path_b = files_a[key], files_b[key]
            meta_a, meta_b = get_image_metadata(path_a), get_image_metadata(path_b)

            if not meta_a or not meta_b:
                continue

            fp_a = ImageFingerprint(path=path_a, hashes=np.array([]), **meta_a)
            fp_b = ImageFingerprint(path=path_b, hashes=np.array([]), **meta_b)

            abs_issues = QCRules.check_absolute(fp_b, self.config)
            rel_issues = QCRules.check_relative(fp_a, fp_b, self.config)
            issues = abs_issues + rel_issues

            if needs_image:
                try:
                    img = load_image(path_b, shrink=4)
                    if img and self.config.qc.check_solid_color and sum(ImageStat.Stat(img).var) < 10.0:
                        issues.append("Solid Color Detected")
                except Exception:
                    pass

            area_a = fp_a.resolution[0] * fp_a.resolution[1]
            area_b = fp_b.resolution[0] * fp_b.resolution[1]

            if self.config.qc.hide_same_resolution_groups and area_a == area_b and not issues:
                processed += 1
                continue

            diff_method = (
                " | ".join(issues)
                if issues
                else ("Same Size" if area_a == area_b else "Higher Resolution")
            )
            best, duplicate = (fp_a, fp_b) if area_a >= area_b else (fp_b, fp_a)
            final_groups[best] = {(duplicate, 100, diff_method)}

            processed += 1
            if processed % 50 == 0:
                self.state.update_progress(processed, len(common_keys))

        return (
            {"db_path": None, "groups_data": final_groups, "lazy_summary": None},
            len(final_groups),
            ScanMode.DUPLICATES,
        )


class SingleFolderQCStrategy(ScanStrategy):
    """
    Strategy for checking a single folder for QC violations.
    """

    def _run_strategy_logic(self, stop_event, start_time, files):
        from collections import defaultdict

        issues_map = defaultdict(list)
        self.state.set_phase("Running QC Checks...", 0.9)
        processed, clean_count = 0, 0

        for path in files:
            if stop_event.is_set():
                break

            meta = get_image_metadata(path)
            if not meta:
                self.all_skipped_files.append(str(path))
                continue

            fp = ImageFingerprint(path=path, hashes=np.array([]), **meta)
            issues = QCRules.check_absolute(fp, self.config)

            if issues:
                issues.sort()
                issues_map[" + ".join(issues)].append(fp)
            else:
                clean_count += 1

            processed += 1
            if processed % 50 == 0:
                self.state.update_progress(processed, len(files))

        self.signals.log_message.emit(
            f"QC Checked: {processed}. Issues: {len(files) - clean_count}. Clean: {clean_count}.",
            "info",
        )

        final_groups = {}
        total_issues = 0
        for issue_name, affected_fps in sorted(issues_map.items()):
            if not affected_fps:
                continue
            total_issues += len(affected_fps)
            header_fp = self._create_qc_header(issue_name, len(affected_fps))
            final_groups[header_fp] = {(fp, 0, "QC Issue") for fp in affected_fps}

        return (
            {"db_path": None, "groups_data": final_groups, "lazy_summary": None},
            total_issues,
            ScanMode.SINGLE_FOLDER_QC,
        )

    def _create_qc_header(self, title: str, count: int) -> ImageFingerprint:
        return ImageFingerprint(
            path=Path(f"ISSUES: {title}"),
            hashes=np.array([]),
            resolution=(0, 0),
            file_size=0,
            mtime=0,
            capture_date=None,
            format_str="QC",
            compression_format="",
            format_details=f"{count} items",
            has_alpha=False,
            bit_depth=0,
            mipmap_count=0,
            texture_type="",
            color_space="",
        )
