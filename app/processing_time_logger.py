"""Daily CSV logging for images processed by the folder watcher."""

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock


CSV_FILE_NAME = "processing_times.csv"
CSV_HEADER = [
    "image_file_name",
    "entered_at",
    "exited_at",
    "total_elapsed_seconds",
]

_write_lock = Lock()


def append_processing_time(
    output_root: Path,
    image_file_name: str,
    entered_at: datetime,
    exited_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Append one successful image-processing record to that day's CSV file."""
    daily_dir = output_root / entered_at.strftime("%Y-%m-%d")
    csv_path = daily_dir / CSV_FILE_NAME

    with _write_lock:
        daily_dir.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(CSV_HEADER)
            writer.writerow([
                image_file_name,
                entered_at.isoformat(timespec="milliseconds"),
                exited_at.isoformat(timespec="milliseconds"),
                f"{elapsed_seconds:.3f}",
            ])

    return csv_path
