from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from . import CONVERSION_VERSION

RECORD_FIELDS = (
    "example_id",
    "source_group_id",
    "dataset_id",
    "decision_type",
    "state",
    "question",
    "candidates",
    "gold_candidate_id",
    "scale_id",
    "ordered_values",
    "score_direction",
)


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Unsupported or invalid registry: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_group_id(dataset_id: str, raw_source_id: Any) -> str:
    return sha256_text(f"{dataset_id}\0{raw_source_id}")


def stable_example_id(dataset_id: str, raw_source_id: Any, variant: str = "main") -> str:
    return sha256_text(f"{dataset_id}\0{raw_source_id}\0{variant}")


def make_candidates(values: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {"id": f"value_{value}", "text": str(value), "order": index}
        for index, value in enumerate(values)
    ]


def make_record(
    *,
    dataset_id: str,
    raw_source_id: Any,
    decision_type: str,
    state: str,
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    gold_candidate_id: str,
    variant: str = "main",
    scale_id: Optional[str] = None,
    ordered_values: Optional[Sequence[int]] = None,
    score_direction: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "example_id": stable_example_id(dataset_id, raw_source_id, variant),
        "source_group_id": stable_group_id(dataset_id, raw_source_id),
        "dataset_id": dataset_id,
        "decision_type": decision_type,
        "state": state,
        "question": question,
        "candidates": [dict(item) for item in candidates],
        "gold_candidate_id": gold_candidate_id,
        "scale_id": scale_id,
        "ordered_values": list(ordered_values or []),
        "score_direction": score_direction,
    }


def make_noul_record(
    *, dataset_id: str, raw_source_id: Any, state: str, question: str, label: str,
    variant: str = "main",
) -> Dict[str, Any]:
    if label not in {"yes", "no"}:
        raise ValueError(f"Binary Noul label must be yes/no, got {label!r}")
    candidates = [{"id": "yes", "text": "Yes", "order": 0}, {"id": "no", "text": "No", "order": 1}]
    return make_record(
        dataset_id=dataset_id,
        raw_source_id=raw_source_id,
        decision_type="noul",
        state=state,
        question=question,
        candidates=candidates,
        gold_candidate_id=label,
        variant=variant,
    )


def make_score_record(
    *, dataset_id: str, raw_source_id: Any, state: str, question: str, score: int,
    scale_id: str, ordered_values: Sequence[int], score_direction: str,
    variant: str = "main",
) -> Dict[str, Any]:
    if score not in ordered_values:
        raise ValueError(f"Score {score} is not valid for {scale_id}")
    candidates = [
        {"id": f"value_{value}", "text": str(value), "order": index}
        for index, value in enumerate(ordered_values)
    ]
    return make_record(
        dataset_id=dataset_id,
        raw_source_id=raw_source_id,
        decision_type="score",
        state=state,
        question=question,
        candidates=candidates,
        gold_candidate_id=f"value_{score}",
        scale_id=scale_id,
        ordered_values=ordered_values,
        score_direction=score_direction,
        variant=variant,
    )


def make_choice_record(
    *, dataset_id: str, raw_source_id: Any, state: str, question: str,
    option_ids: Sequence[str], option_texts: Sequence[str], gold_index: int,
    variant: str = "main",
) -> Dict[str, Any]:
    if len(option_ids) != len(option_texts) or not option_ids:
        raise ValueError("Choice options must have matching non-empty IDs and texts")
    if gold_index < 0 or gold_index >= len(option_ids):
        raise ValueError(f"Invalid choice target index: {gold_index}")
    candidates = [
        {"id": option_id, "text": text, "order": index}
        for index, (option_id, text) in enumerate(zip(option_ids, option_texts))
    ]
    return make_record(
        dataset_id=dataset_id,
        raw_source_id=raw_source_id,
        decision_type="choice",
        state=state,
        question=question,
        candidates=candidates,
        gold_candidate_id=option_ids[gold_index],
        variant=variant,
    )


def validate_records(records: Sequence[Mapping[str, Any]], expected_count: Optional[int] = None) -> None:
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, got {len(records)}")
    seen = set()
    for index, record in enumerate(records):
        missing = set(RECORD_FIELDS) - set(record)
        if missing:
            raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")
        if record["example_id"] in seen:
            raise ValueError(f"Duplicate example_id: {record['example_id']}")
        seen.add(record["example_id"])
        if record["decision_type"] not in {"choice", "noul", "score"}:
            raise ValueError(f"Unknown decision type {record['decision_type']!r}")
        if not record["state"].strip() or not record["question"].strip():
            raise ValueError(f"Record {index} has an empty state or question")
        ids = [candidate["id"] for candidate in record["candidates"]]
        if len(ids) != len(set(ids)) or record["gold_candidate_id"] not in ids:
            raise ValueError(f"Invalid candidates/target in record {record['example_id']}")
        if record["decision_type"] == "noul" and set(ids) != {"yes", "no"}:
            raise ValueError("Noul records must contain exactly the yes/no candidates")
        if record["decision_type"] == "score":
            if not record["scale_id"] or not record["score_direction"]:
                raise ValueError("Score records require scale_id and score_direction")
            values = record["ordered_values"]
            candidate_values = [int(candidate["text"]) for candidate in record["candidates"]]
            if list(values) != candidate_values:
                raise ValueError(f"Score candidate order does not match ordered_values in {record['example_id']}")
        elif record["scale_id"] is not None or record["ordered_values"] or record["score_direction"] is not None:
            raise ValueError("Only score records may contain scale metadata")


def split_records_exact(records: Sequence[Dict[str, Any]], counts: Mapping[str, int], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    requested = sum(counts.values())
    if len(records) < requested:
        raise ValueError(f"Need {requested} source rows, found {len(records)}")
    keys = list(range(len(records)))
    random.Random(seed).shuffle(keys)
    chosen = keys[:requested]
    result: Dict[str, List[Dict[str, Any]]] = {}
    offset = 0
    for split_name, count in counts.items():
        result[split_name] = [records[index] for index in chosen[offset:offset + count]]
        offset += count
    assert_group_disjoint(result)
    return result


def stratified_split(records: Sequence[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """Split one-record-per-source groups 60/20/20, preserving label proportions."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(record["gold_candidate_id"], []).append(record)
    output = {"train": [], "validation": [], "test": []}
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * 0.6)
        n_validation = int(n * 0.2)
        output["train"].extend(rows[:n_train])
        output["validation"].extend(rows[n_train:n_train + n_validation])
        output["test"].extend(rows[n_train + n_validation:])
    for rows in output.values():
        rows.sort(key=lambda row: row["example_id"])
    assert_group_disjoint(output)
    return output


def stratified_split_exact(
    records: Sequence[Dict[str, Any]],
    counts_by_label: Mapping[str, Mapping[str, int]],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split records to exact per-label counts while keeping source groups separate."""
    split_names = list(next(iter(counts_by_label.values())).keys()) if counts_by_label else []
    output: Dict[str, List[Dict[str, Any]]] = {name: [] for name in split_names}
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record["gold_candidate_id"]), []).append(record)
    if set(by_label) != set(counts_by_label):
        raise ValueError(f"Labels differ from requested stratification: found {sorted(by_label)}, requested {sorted(counts_by_label)}")
    for label_index, label in enumerate(sorted(by_label)):
        label_counts = counts_by_label[label]
        if list(label_counts) != split_names:
            raise ValueError(f"Split names differ for label {label!r}")
        pieces = split_records_exact(by_label[label], label_counts, seed + label_index)
        for split_name, rows in pieces.items():
            output[split_name].extend(rows)
    for rows in output.values():
        rows.sort(key=lambda row: row["example_id"])
    assert_group_disjoint(output)
    return output


def assert_group_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    owner: Dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            group_id = row["source_group_id"]
            previous = owner.get(group_id)
            if previous is not None and previous != split_name:
                raise ValueError(f"Source group appears in both {previous} and {split_name}")
            owner[group_id] = split_name


def _arrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("Parquet output requires the pinned dependencies in requirements-data.txt") from exc
    candidate = pa.struct([
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("order", pa.int32(), nullable=False),
    ])
    return pa.schema([
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("source_group_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("decision_type", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("candidates", pa.list_(candidate), nullable=False),
        pa.field("gold_candidate_id", pa.string(), nullable=False),
        pa.field("scale_id", pa.string()),
        pa.field("ordered_values", pa.list_(pa.int32()), nullable=False),
        pa.field("score_direction", pa.string()),
    ])


def records_checksum(records: Iterable[Mapping[str, Any]], source_revision: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_revision.encode("utf-8"))
    digest.update(b"\n")
    for record in sorted(records, key=lambda item: item["example_id"]):
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _record_signature(records: Iterable[Mapping[str, Any]]) -> str:
    return records_checksum(records, CONVERSION_VERSION)


def write_splits(
    *,
    splits: Mapping[str, Sequence[Dict[str, Any]]],
    output_dir: Path,
    dataset_id: str,
    seed: int,
    source_revision: str,
    source_checksum: Optional[str] = None,
    source_checksum_method: str = "sha256_of_selected_example_ids_and_labels",
    rejected_row_counts: Optional[Mapping[str, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    materialized = {name: list(rows) for name, rows in splits.items()}
    for rows in materialized.values():
        validate_records(rows)
    assert_group_disjoint(materialized)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_name in ("development", "calibration", "internal_holdout", "external_validation"):
        (output_dir / f"{obsolete_name}.parquet").unlink(missing_ok=True)
    schema = _arrow_schema()
    for split_name, rows in materialized.items():
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, output_dir / f"{split_name}.parquet", compression="zstd")
    all_records = [record for rows in materialized.values() for record in rows]
    manifest = {
        "manifest_schema_version": 1,
        "dataset_id": dataset_id,
        "seed": seed,
        "source_revision": source_revision,
        "source_checksum": source_checksum or _record_signature(all_records),
        "source_checksum_method": source_checksum_method,
        "conversion_version": CONVERSION_VERSION,
        "split_counts": {name: len(rows) for name, rows in materialized.items()},
        "rejected_row_counts": dict(rejected_row_counts or {}),
        "hashed_source_ids": {
            name: sorted({record["source_group_id"] for record in rows})
            for name, rows in materialized.items()
        },
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_from_nested(row: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    value: Any = row
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value
