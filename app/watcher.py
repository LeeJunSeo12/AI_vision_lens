from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from threading import Event, Thread
import time
from typing import Any
from zoneinfo import ZoneInfo

from app import config
from app.inference import IMG_EXTS, InvalidImageError, LensDefectPipeline
from app.processing_time_logger import append_processing_time


FileSignature = tuple[int, int]
KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


class ImageFolderWatcher:
    def __init__(self, pipeline: LensDefectPipeline):
        self.pipeline = pipeline
        self.enabled = config.WATCH_ENABLED
        self.input_dir = Path(config.WATCH_INPUT_DIR)
        self.output_dir = Path(config.WATCH_OUTPUT_DIR)
        self.poll_interval = float(config.WATCH_POLL_INTERVAL_SECONDS)
        self.file_stable_seconds = float(config.WATCH_FILE_STABLE_SECONDS)
        self.process_existing_on_start = bool(config.WATCH_PROCESS_EXISTING_ON_START)

        self._seen: dict[Path, FileSignature] = {}
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._processed_count = 0
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.enabled:
            print("folder watcher disabled", flush=True)
            return

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.process_existing_on_start:
            self._seen = self._snapshot_existing_files()

        self._thread = Thread(target=self._run, name="image-folder-watcher", daemon=True)
        self._thread.start()
        print(
            f"folder watcher started. input={self.input_dir}, output={self.output_dir}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self._thread is not None and self._thread.is_alive(),
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "processed_count": self._processed_count,
            "last_result": self._last_result,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as exc:
                self._last_error = str(exc)
                print(f"folder watcher error: {exc}", flush=True)

            self._stop_event.wait(self.poll_interval)

    def _scan_once(self) -> None:
        for path in self._iter_image_files():
            signature = self._signature(path)
            if signature is None or self._seen.get(path) == signature:
                continue

            if not self._is_stable(path):
                continue

            try:
                self._process_file(path)
                self._seen[path] = signature
            except InvalidImageError as exc:
                self._last_error = f"{path.name}: {exc}"
                self._seen[path] = signature
                print(f"invalid image skipped: {path} ({exc})", flush=True)
            except OSError as exc:
                self._last_error = f"{path.name}: {exc}"
                print(f"file process failed: {path} ({exc})", flush=True)

    def _iter_image_files(self) -> list[Path]:
        if not self.input_dir.exists():
            return []

        return sorted(
            path
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        )

    def _snapshot_existing_files(self) -> dict[Path, FileSignature]:
        snapshot: dict[Path, FileSignature] = {}
        for path in self._iter_image_files():
            signature = self._signature(path)
            if signature is not None:
                snapshot[path] = signature
        return snapshot

    def _signature(self, path: Path) -> FileSignature | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def _is_stable(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False

        if stat.st_size <= 0:
            return False

        age_seconds = time.time() - stat.st_mtime
        return age_seconds >= self.file_stable_seconds

    def _process_file(self, path: Path) -> None:
        entered_at = datetime.now(KOREA_TIMEZONE)
        start_time = time.perf_counter()

        image_bytes = path.read_bytes()
        result = self.pipeline.predict_bytes(image_bytes, file_name=path.name)

        okng = "NG" if result["pred_okng"] == "NG" else "OK"
        output_path = self._output_path(path, okng, entered_at)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output_path)

        exited_at = datetime.now(KOREA_TIMEZONE)
        elapsed_seconds = time.perf_counter() - start_time
        csv_path = append_processing_time(
            output_root=self.output_dir,
            image_file_name=path.name,
            entered_at=entered_at,
            exited_at=exited_at,
            elapsed_seconds=elapsed_seconds,
        )

        self._processed_count += 1
        self._last_result = {
            "source": str(path),
            "saved_to": str(output_path),
            "processing_time_csv": str(csv_path),
            "model1_pred": result.get("model1_pred"),
            "final_folder": result.get("final_folder"),
            "pred_okng": okng,
            "result": result.get("result"),
        }
        self._last_error = None
        print(f"image processed: {path.name} -> {output_path}", flush=True)

    def _output_path(self, source_path: Path, okng: str, processed_at: datetime) -> Path:
        date_dir = processed_at.strftime("%Y-%m-%d")
        file_name = f"{source_path.stem}_{okng}{source_path.suffix}"
        return self.output_dir / date_dir / file_name
