# app/workflow/strategies.py
"""
Contains different strategies for the scanning process.
Refactored to use DB_SERVICE singleton, updated PipelineManager signature,
and ensure results are persisted for UI access.
"""

import contextlib
import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from app.ai.manager import init_worker, worker_get_single_vector, worker_get_text_vector
from app.domain.data_models import (
    AnalysisItem,
    DuplicateResults,
    EvidenceMethod,
    ImageFingerprint,
    ScanConfig,
    ScanMode,
    ScanState,
)
from app.imaging.image_io import get_image_metadata, load_image
from app.infrastructure.db_service import DB_SERVICE
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

app_logger = logging.getLogger("PixelHand.workflow.strategies")


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: SignalBus, scanner_core):
        self.config = config
        self.state = state
        self.signals = signals
        self.scanner_core = scanner_core
        self.all_skipped_files: list[str] = []

    @abstractmethod
    def execute(self, stop_event: threading.Event, start_time: float):
        pass

    def _find_files_as_list(self, stop_event: threading.Event) -> list[Path]:
        self.state.set_phase("Finding image files...", 0.1)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files = [path for batch in finder.stream_files(stop_event) if not stop_event.is_set() for path, _ in batch]
        files.sort()
        return files

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        try:
            meta = get_image_metadata(path)
            if meta:
                return ImageFingerprint(path=path, hashes=np.array([]), **meta)
        except Exception:
            self.all_skipped_files.append(str(path))
        return None


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate images using a granular pipeline of stages."""

    def execute(self, stop_event: threading.Event, start_time: float):
        # FIX: Use DB_SERVICE instead of self.scanner_core.db_context
        context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            lancedb_db=DB_SERVICE.db,
            lancedb_table=DB_SERVICE.table,
        )

        pipeline_to_run = self._build_pipeline()

        for stage, weight in pipeline_to_run:
            if stop_event.is_set():
                break
            context.state.set_phase(stage.name, weight)
            if not stage.run(context):
                break

        self.all_skipped_files.extend(context.all_skipped_files)

        if not stop_event.is_set():
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
            self._report_and_cleanup(final_groups, start_time)
        else:
            self._report_and_cleanup({}, start_time)

    def _build_pipeline(self) -> list[tuple[ScanStage, float]]:
        pipeline = [(FileDiscoveryStage(), 0.05)]
        if self.config.find_exact_duplicates:
            pipeline.append((ExactDuplicateStage(), 0.10))

        needs_items = (
            self.config.find_simple_duplicates
            or self.config.find_perceptual_duplicates
            or self.config.find_structural_duplicates
            or self.config.use_ai
        )
        if needs_items:
            pipeline.append((ItemGenerationStage(), 0.05))

        if any(
            [
                self.config.find_simple_duplicates,
                self.config.find_perceptual_duplicates,
                self.config.find_structural_duplicates,
            ]
        ):
            pipeline.append((PerceptualDuplicateStage(), 0.20))

        if self.config.use_ai:
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

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        num_found = sum(len(d) for d in final_groups.values())
        duration = time.time() - start_time
        # FIX: Access DB URI via DB_SERVICE
        db_uri = str(DB_SERVICE.db.uri) if (LANCEDB_AVAILABLE and DB_SERVICE.db) else None
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
                app_logger.error(f"Failed to persist results: {e}", exc_info=True)
                self.signals.scan_error.emit(f"Result persistence failed: {e}")
                return

        payload = {"db_path": db_uri, "groups_data": None, "lazy_summary": groups_summary}
        self.scanner_core._finalize_scan(payload, num_found, ScanMode.DUPLICATES, duration, self.all_skipped_files)

    def _persist_results_to_lancedb(self, final_groups: DuplicateResults):
        data = []
        group_id = 1
        for best_fp, dups in final_groups.items():
            row = best_fp.to_lancedb_dict()
            row.update({"group_id": group_id, "is_best": True, "distance": 0, "found_by": BEST_FILE_METHOD_NAME})
            if "vector" in row:
                del row["vector"]
            if "id" in row:
                del row["id"]
            data.append(row)

            for dup_fp, score, method in dups:
                row = dup_fp.to_lancedb_dict()
                row.update({"group_id": group_id, "is_best": False, "distance": score, "found_by": method})
                if "vector" in row:
                    del row["vector"]
                if "id" in row:
                    del row["id"]
                data.append(row)
            group_id += 1

        if not data:
            return

        # FIX: Use DB_SERVICE to save the results table thread-safely
        DB_SERVICE.save_results_table(data)


class SearchStrategy(ScanStrategy):
    """Strategy for text- or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        all_files = self._find_files_as_list(stop_event)

        # Check if we found files
        if not all_files:
            self.signals.log_message.emit("No files found in the selected folder.", "warning")
            self.scanner_core._finalize_scan(None, 0, self.config.scan_mode, 0, [])
            return

        if self.scanner_core._check_stop_or_empty(
            stop_event, all_files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        ):
            return

        self.state.set_phase("Phase 2/2: Creating AI fingerprints...", 0.8)

        # FIX: PipelineManager no longer needs vector_db_writer passed explicitly
        pipeline_manager = PipelineManager(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
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
            lancedb_db=DB_SERVICE.db,
            lancedb_table=DB_SERVICE.table,
            items_to_process=items,
            all_image_fps=fps_map,
        )

        success, skipped = pipeline_manager.run(search_context)
        self.all_skipped_files.extend(skipped)

        if not success and not stop_event.is_set():
            self.signals.scan_error.emit("Failed to generate fingerprints.")
            return

        # Force a small wait to ensure LanceDB flushes, though DB_SERVICE should handle it
        if not stop_event.is_set():
            # Check if DB has data
            try:
                count = DB_SERVICE.table.to_lance().count_rows()
                self.signals.log_message.emit(f"Database contains {count} vectors.", "info")
                if count == 0:
                    self.signals.log_message.emit("No vectors were generated. Check model/files.", "warning")
                    self.scanner_core._finalize_scan(
                        None, 0, self.config.scan_mode, time.time() - start_time, self.all_skipped_files
                    )
                    return
            except Exception as e:
                app_logger.warning(f"Could not count rows: {e}")

        num_found, db_path, results_list = self._perform_similarity_search()

        if num_found is None:
            # Error occurred during search
            return

        final_groups = self._create_search_groups(results_list)

        # IMPORTANT: Persist these results to 'scan_results' table so UI can read them
        if num_found > 0:
            self._persist_search_results(final_groups)

        self.scanner_core._finalize_scan(
            {"db_path": db_path, "groups_data": final_groups, "lazy_summary": None},
            num_found,
            self.config.scan_mode,
            time.time() - start_time,
            self.all_skipped_files,
        )

    def _perform_similarity_search(self) -> tuple[int | None, str | None, list[tuple[ImageFingerprint, float]]]:
        self.state.set_phase("Searching for similar images...", 0.1)
        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.scan_error.emit("Could not generate a vector for the search query.")
            return None, None, []

        dist_threshold = 1.0 - (self.config.similarity_threshold / 100.0)
        results = []

        if LANCEDB_AVAILABLE and DB_SERVICE.is_ready:
            try:
                from app.shared.constants import DEFAULT_SEARCH_PRECISION, SEARCH_PRECISION_PRESETS

                precision_config = SEARCH_PRECISION_PRESETS.get(
                    self.config.search_precision, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
                )

                # FIX: Use DB_SERVICE to search
                raw_rows = DB_SERVICE.search_vectors(
                    query_vector,
                    limit=SIMILARITY_SEARCH_K_NEIGHBORS,
                    probes=precision_config["nprobes"],
                    refine=precision_config["refine_factor"],
                )

                # Filter results by distance threshold manually since search_vectors returns list[dict]
                self.signals.log_message.emit(f"Raw search returned {len(raw_rows)} candidates.", "info")

                for row_dict in raw_rows:
                    dist = row_dict["_distance"]
                    if dist < dist_threshold:
                        results.append((ImageFingerprint.from_db_row(row_dict), dist))

                self.signals.log_message.emit(
                    f"Filtered to {len(results)} matches (threshold {dist_threshold}).", "info"
                )

            except Exception as e:
                self.signals.log_message.emit(f"LanceDB search failed: {e}", "error")
                return None, None, []
        else:
            self.signals.scan_error.emit("No LanceDB is available.")
            return None, None, []

        return len(results), str(DB_SERVICE.db.uri), results

    def _create_search_groups(self, results: list[tuple[ImageFingerprint, float]]) -> DuplicateResults:
        best_fp = self._create_search_context_fingerprint()
        if not best_fp:
            return {}
        dups = set()
        for fp, d in results:
            score = int(max(0.0, (1.0 - d) * 100))
            dups.add((fp, score, EvidenceMethod.AI.value))
        return {best_fp: dups}

    def _persist_search_results(self, final_groups: DuplicateResults):
        """
        Saves search results to the 'scan_results' table so the UI can fetch them via DB_SERVICE.
        """
        data = []
        group_id = 1  # Search usually has only one group (the query)
        for best_fp, dups in final_groups.items():
            # Note: The query itself (best_fp) might not be in the DB if it's text,
            # but we still save it as the group leader.
            row = best_fp.to_lancedb_dict()
            row.update({"group_id": group_id, "is_best": True, "distance": 0, "found_by": "Query"})
            # Remove vector to save space in results table
            if "vector" in row:
                del row["vector"]
            if "id" in row:
                del row["id"]
            data.append(row)

            for dup_fp, score, method in dups:
                row = dup_fp.to_lancedb_dict()
                row.update({"group_id": group_id, "is_best": False, "distance": score, "found_by": method})
                if "vector" in row:
                    del row["vector"]
                if "id" in row:
                    del row["id"]
                data.append(row)
            group_id += 1

        if data:
            DB_SERVICE.save_results_table(data)

    def _create_search_context_fingerprint(self) -> ImageFingerprint | None:
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return self._create_dummy_fp(self.config.sample_path)
        else:
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

    def _get_query_vector(self) -> np.ndarray | None:
        # Re-init worker to ensure model is ready for single inference
        init_worker({"model_name": self.config.model_name, "device": self.config.device, "threads_per_worker": 1})

        if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
            self.signals.log_message.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
            return worker_get_text_vector(self.config.search_query)

        elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            self.signals.log_message.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
            return worker_get_single_vector(str(self.config.sample_path))
        return None


class FolderComparisonStrategy(ScanStrategy):
    def execute(self, stop_event: threading.Event, start_time: float):
        folder_a = self.config.folder_path
        folder_b = self.config.comparison_folder_path
        if not folder_b:
            self.signals.scan_error.emit("Secondary folder not specified.")
            self.scanner_core._finalize_scan(None, 0, None, 0, [])
            return

        def get_key(p: Path):
            return p.stem.lower() if self.config.match_by_stem else p.name.lower()

        self.state.set_phase(f"Scanning Folder A: {folder_a.name}...", 0.1)
        finder_a = FileFinder(
            self.state, folder_a, self.config.excluded_folders, self.config.selected_extensions, self.signals
        )
        files_a = {get_key(p): p for batch in finder_a.stream_files(stop_event) for p, _ in batch}
        if stop_event.is_set():
            return

        self.state.set_phase(f"Scanning Folder B: {folder_b.name}...", 0.1)
        finder_b = FileFinder(
            self.state, folder_b, self.config.excluded_folders, self.config.selected_extensions, self.signals
        )
        files_b = {get_key(p): p for batch in finder_b.stream_files(stop_event) for p, _ in batch}
        if stop_event.is_set():
            return

        common_keys = set(files_a.keys()) & set(files_b.keys())
        self.state.set_phase("Comparing metadata & QC...", 0.8)
        final_groups: DuplicateResults = {}
        processed = 0

        ImageStat = None
        if self.config.qc_check_solid_color:
            with contextlib.suppress(ImportError):
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

            # --- Image Loading for Heavy Checks ---
            img = None
            needs_image = False

            # Check 1: Solid Color
            if self.config.qc_check_solid_color:
                needs_image = True

            # Check 2: Normal Maps (With optional tag filtering)
            check_normals = self.config.qc_check_normal_maps
            if check_normals and self.config.qc_normal_maps_tags:
                fname = path_b.name.lower()
                # If tags list is NOT empty and filename contains NO tags -> skip check
                if not any(t in fname for t in self.config.qc_normal_maps_tags):
                    check_normals = False

            if check_normals:
                needs_image = True

            if needs_image:
                try:
                    # Smart shrink: Don't shrink too much for Normal Map check (min 512px preferred)
                    # For solid color we can shrink a lot.
                    shrink = max(1, min(fp_b.resolution) // 32) if not check_normals else 4
                    img = load_image(path_b, shrink=shrink)
                except Exception:
                    pass

            if img:
                # Solid Color Logic
                if self.config.qc_check_solid_color and ImageStat:
                    try:
                        if sum(ImageStat.Stat(img).var) < 10.0:
                            issues.append("Solid Color Detected")
                    except Exception:
                        pass

                # Normal Map Logic
                if check_normals:
                    nm_res = QCRules.check_normal_map_integrity(img)
                    if nm_res:
                        group_name, detail = nm_res
                        issues.append(group_name)
                        fp_b.format_details += f" [NM Err: {detail}]"

            area_a = fp_a.resolution[0] * fp_a.resolution[1]
            area_b = fp_b.resolution[0] * fp_b.resolution[1]

            if self.config.hide_same_resolution_groups and area_a == area_b and not issues:
                processed += 1
                continue

            diff_method = " | ".join(issues) if issues else ("Same Size" if area_a == area_b else "Higher Resolution")
            best, duplicate = (fp_a, fp_b) if area_a >= area_b else (fp_b, fp_a)
            final_groups[best] = {(duplicate, 100, diff_method)}
            processed += 1
            if processed % 50 == 0:
                self.state.update_progress(processed, len(common_keys))

        self.scanner_core._finalize_scan(
            {"db_path": None, "groups_data": final_groups, "lazy_summary": None},
            len(final_groups),
            ScanMode.DUPLICATES,
            time.time() - start_time,
            self.all_skipped_files,
        )


class SingleFolderQCStrategy(ScanStrategy):
    def execute(self, stop_event: threading.Event, start_time: float):
        files = self._find_files_as_list(stop_event)
        if self.scanner_core._check_stop_or_empty(
            stop_event, files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        ):
            return

        from collections import defaultdict

        issues_map = defaultdict(list)
        self.state.set_phase("Running QC Checks...", 0.9)
        processed, clean_count = 0, 0

        ImageStat = None
        if self.config.qc_check_solid_color:
            with contextlib.suppress(ImportError):
                from PIL import ImageStat

        for path in files:
            if stop_event.is_set():
                break
            meta = get_image_metadata(path)
            if not meta:
                self.all_skipped_files.append(str(path))
                continue

            fp = ImageFingerprint(path=path, hashes=np.array([]), **meta)
            issues = QCRules.check_absolute(fp, self.config)

            # --- Image Loading for Heavy Checks ---
            img = None
            needs_image = False

            # Check 1: Solid Color
            if self.config.qc_check_solid_color:
                needs_image = True

            # Check 2: Normal Maps (With optional tag filtering)
            check_normals = self.config.qc_check_normal_maps
            if check_normals and self.config.qc_normal_maps_tags:
                fname = path.name.lower()
                # If tags list is NOT empty and filename contains NO tags -> skip check
                if not any(t in fname for t in self.config.qc_normal_maps_tags):
                    check_normals = False

            if check_normals:
                needs_image = True

            if needs_image:
                try:
                    # Smart shrink
                    shrink = max(1, min(fp.resolution) // 32) if not check_normals else 4
                    img = load_image(path, shrink=shrink)
                except Exception:
                    pass

            # Apply Checks
            if img:
                # Solid Color Logic
                if self.config.qc_check_solid_color and ImageStat:
                    try:
                        if sum(ImageStat.Stat(img).var) < 10.0:
                            issues.append("Solid Color Texture")
                    except Exception:
                        pass

                # Normal Map Logic
                if check_normals:
                    nm_res = QCRules.check_normal_map_integrity(img)
                    if nm_res:
                        group_name, detail = nm_res
                        issues.append(group_name)
                        fp.format_details += f" [NM Err: {detail}]"

            if issues:
                issues.sort()
                issues_map[" + ".join(issues)].append(fp)
            else:
                clean_count += 1
            processed += 1
            if processed % 50 == 0:
                self.state.update_progress(processed, len(files))

        self.signals.log_message.emit(
            f"QC Checked: {processed}. Issues: {len(files) - clean_count - len(self.all_skipped_files)}. Clean: {clean_count}.",
            "info",
        )
        if stop_event.is_set():
            return

        final_groups = {}
        total_issues = 0
        for issue_name, affected_fps in sorted(issues_map.items()):
            if not affected_fps:
                continue
            total_issues += len(affected_fps)
            header_fp = self._create_qc_header(issue_name, len(affected_fps))
            final_groups[header_fp] = {(fp, 0, "QC Issue") for fp in affected_fps}

        self.scanner_core._finalize_scan(
            {"db_path": None, "groups_data": final_groups, "lazy_summary": None},
            total_issues,
            ScanMode.SINGLE_FOLDER_QC,
            time.time() - start_time,
            self.all_skipped_files,
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
