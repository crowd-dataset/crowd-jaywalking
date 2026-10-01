"""Create deterministic stratified JAAD development, validation, and locked test splits."""

from __future__ import annotations

import csv
import random

from scripts.core.config import ProjectConfig
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    source = config.data_file("source_annotations")
    destination = config.data_file("annotations")

    development_fraction = float(config.get("development_fraction"))
    validation_fraction = float(config.get("validation_fraction"))
    locked_fraction = float(config.get("locked_test_fraction"))
    if abs(development_fraction + validation_fraction + locked_fraction - 1.0) > 1e-9:
        raise ValueError("Development, validation, and locked test fractions must sum to 1.0")
    if source == destination:
        raise ValueError("Source annotations and split output must be different files")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not {"filename", "label"}.issubset(fieldnames):
        raise ValueError("Source annotations must contain filename and label columns")

    groups: dict[str, list[dict[str, str]]] = {"yes": [], "no": []}
    excluded: list[dict[str, str]] = []
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label in groups:
            groups[label].append(row)
        else:
            excluded.append(row)

    generator = random.Random(int(config.get("split_seed")))
    output_rows: list[dict[str, str]] = []
    counts = {"development": 0, "validation": 0, "locked_test": 0, "excluded": 0}

    for label in ("yes", "no"):
        group = list(groups[label])
        generator.shuffle(group)
        total = len(group)
        validation_count = int(round(total * validation_fraction))
        locked_count = int(round(total * locked_fraction))
        if total >= 5:
            validation_count = max(1, validation_count)
            locked_count = max(1, locked_count)
        if validation_count + locked_count >= total:
            validation_count = max(0, total // 5)
            locked_count = max(0, total // 5)

        for index, row in enumerate(group):
            updated = dict(row)
            if index < locked_count:
                split = "locked_test"
            elif index < locked_count + validation_count:
                split = "validation"
            else:
                split = "development"
            updated["split"] = split
            output_rows.append(updated)
            counts[split] += 1

    for row in excluded:
        updated = dict(row)
        updated["split"] = "excluded"
        output_rows.append(updated)
        counts["excluded"] += 1

    output_rows.sort(key=lambda row: str(row.get("filename", "")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [name for name in fieldnames if name != "split"] + ["split"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info("Saved: {}", destination)
    logger.info(
        "Development={} Validation={} LockedTest={} Excluded={}",
        counts['development'],
        counts['validation'],
        counts['locked_test'],
        counts['excluded'],
    )


if __name__ == "__main__":
    main()
