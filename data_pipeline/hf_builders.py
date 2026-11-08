from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .core import (
    load_registry,
    assert_group_disjoint,
    make_choice_record,
    make_noul_record,
    split_records_exact,
    stratified_split_exact,
    records_checksum,
    value_from_nested,
    write_splits,
)


def _load_dataset(repo: str, config: str, revision: str, split: str = "train"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-data.txt to download Hugging Face datasets") from exc
    return load_dataset(repo, config, split=split, revision=revision)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _medmcqa_label(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"a", "b", "c", "d"}:
            return ord(value) - ord("a")
        value = int(value)
    value = int(value)
    if value not in range(4):
        raise ValueError(f"Invalid MedMCQA zero-based answer index: {value}")
    return value


def _convert_medmcqa_row(row: Mapping[str, Any], conversion: Mapping[str, Any]) -> Dict[str, Any]:
    source_id = row[conversion["source_id_field"]]
    question = _clean_text(row[conversion["question_field"]])
    options = [_clean_text(row[field]) for field in conversion["option_fields"]]
    label = _medmcqa_label(row[conversion["label_field"]])
    if not question or any(not option for option in options):
        raise ValueError("empty question or option")
    metadata = []
    for field in conversion.get("metadata_fields", []):
        value = _clean_text(row.get(field))
        if value:
            metadata.append(f"{field.replace('_', ' ').title()}: {value}")
    state = "\n".join(metadata + [f"Question: {question}"])
    return make_choice_record(
        dataset_id="medmcqa",
        raw_source_id=source_id,
        state=state,
        question="Select the single best answer.",
        option_ids=conversion["option_ids"],
        option_texts=options,
        gold_index=label,
    )


def build_medmcqa(registry_path: Path, seed: int = 42) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], str]:
    registry = load_registry(registry_path)
    source = registry["source"]
    conversion = registry["conversion"]
    allocation = registry["internal_allocation"]
    train_split_name = allocation["from_source_split"]
    expected_train = registry["source_splits"][train_split_name]["expected_rows"]
    train_data = _load_dataset(source["repository"], source.get("config", "default"), source["revision"], split="train")
    if len(train_data) != expected_train:
        raise ValueError(f"Pinned MedMCQA train split has {len(train_data)} rows; registry expects {expected_train}")
    expected_validation = registry["source_splits"]["validation"]["expected_rows"]
    validation_data = _load_dataset(source["repository"], source.get("config", "default"), source["revision"], split="validation")
    if len(validation_data) != expected_validation:
        raise ValueError(f"Pinned MedMCQA validation split has {len(validation_data)} rows; registry expects {expected_validation}")

    counts = {name: int(count) for name, count in allocation["counts"].items()}
    required = sum(counts.values())
    shuffled = train_data.shuffle(seed=seed)
    records: List[Dict[str, Any]] = []
    rejected = {"train_missing_or_invalid_fields": 0, "test_missing_or_invalid_fields": 0}
    for row in shuffled:
        try:
            records.append(_convert_medmcqa_row(row, conversion))
        except (KeyError, TypeError, ValueError):
            rejected["train_missing_or_invalid_fields"] += 1
        if len(records) == required:
            break
    if len(records) != required:
        raise ValueError(f"Could only build {len(records)} valid MedMCQA train records; need {required}")
    splits = split_records_exact(records, counts, seed)

    test_count = int(allocation["external_test_count"])
    external_rows = validation_data.shuffle(seed=seed)
    test_records: List[Dict[str, Any]] = []
    for row in external_rows:
        try:
            test_records.append(_convert_medmcqa_row(row, conversion))
        except (KeyError, TypeError, ValueError):
            rejected["test_missing_or_invalid_fields"] += 1
        if len(test_records) == test_count:
            break
    if len(test_records) != test_count:
        raise ValueError(f"Could only build {len(test_records)} valid MedMCQA test records; need {test_count}")
    splits["test"] = test_records
    assert_group_disjoint(splits)
    checksum = records_checksum((row for part in splits.values() for row in part), source["revision"])
    return splits, rejected, checksum

def _normalize_pubmed_label(value: Any) -> str:
    label = str(value).strip().lower()
    aliases = {"true": "yes", "false": "no"}
    return aliases.get(label, label)


def _pubmed_record(row: Mapping[str, Any], dataset_id: str, label: str) -> Dict[str, Any]:
    source_id = row.get("pubid")
    question_text = _clean_text(row.get("question"))
    contexts = value_from_nested(row, "context.contexts", [])
    if isinstance(contexts, str):
        contexts = [contexts]
    context_text = "\n".join(_clean_text(item) for item in (contexts or []) if _clean_text(item))
    if not source_id or not question_text or not context_text:
        raise ValueError("missing PubMedQA article ID, question, or study context")
    state = f"Research question: {question_text}\nStudy context: {context_text}"
    return make_noul_record(
        dataset_id=dataset_id,
        raw_source_id=source_id,
        state=state,
        question="Does the provided study context support the proposition in the research question?",
        label=label,
    )


def build_pubmedqa(registry_path: Path, seed: int = 42) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], str]:
    registry = load_registry(registry_path)
    source = registry["source"]
    conversion = registry["conversion"]
    try:
        artificial = _load_dataset(source["repository"], "pqa_artificial", source["revision"])
        labeled = _load_dataset(source["repository"], "pqa_labeled", source["revision"])
    except Exception as exc:
        raise RuntimeError(f"Could not load pinned PubMedQA configs: {exc}") from exc
    for config_name, data in (("pqa_artificial", artificial), ("pqa_labeled", labeled)):
        expected_rows = registry["source_splits"][config_name]["expected_rows"]
        if len(data) != expected_rows:
            raise ValueError(f"Pinned PubMedQA {config_name} config has {len(data)} rows; registry expects {expected_rows}")

    rejected = {
        "artificial_missing_or_invalid": 0,
        "artificial_duplicate_pubid": 0,
        "cross_split_article_overlap_excluded": 0,
        "artificial_excluded_maybe_or_other": 0,
    }
    configured_splits = registry["training_allocation"]["split_counts"]
    split_names = list(configured_splits)
    per_label_counts = {
        label: {split_name: int(configured_splits[split_name][label]) for split_name in split_names}
        for label in ("yes", "no")
    }
    wanted = {label: sum(counts.values()) for label, counts in per_label_counts.items()}
    labeled_pubids = {str(row.get("pubid")) for row in labeled if row.get("pubid")}
    by_label: Dict[str, List[Dict[str, Any]]] = {"yes": [], "no": []}
    seen = set()
    shuffled = artificial.shuffle(seed=seed)
    for row in shuffled:
        label = _normalize_pubmed_label(row.get(conversion["label_field"]))
        if label not in by_label:
            rejected["artificial_excluded_maybe_or_other"] += 1
            continue
        pubid = row.get(conversion["source_id_field"])
        if not pubid:
            rejected["artificial_missing_or_invalid"] += 1
            continue
        if str(pubid) in labeled_pubids:
            rejected["cross_split_article_overlap_excluded"] += 1
            continue
        if str(pubid) in seen:
            rejected["artificial_duplicate_pubid"] += 1
            continue
        try:
            record = _pubmed_record(row, "pubmedqa", label)
        except (TypeError, ValueError):
            rejected["artificial_missing_or_invalid"] += 1
            continue
        seen.add(str(pubid))
        if len(by_label[label]) < wanted[label]:
            by_label[label].append(record)
        if all(len(by_label[key]) >= wanted[key] for key in wanted):
            break
    for label, count in wanted.items():
        if len(by_label[label]) != count:
            raise ValueError(f"PQA-A has only {len(by_label[label])} clean unique {label} records; need {count}")
    splits = stratified_split_exact(
        by_label["yes"] + by_label["no"],
        per_label_counts,
        seed=seed,
    )
    checksum = records_checksum((row for rows in splits.values() for row in rows), source["revision"])
    return splits, rejected, checksum


def build_and_write_medmcqa(registry_path: Path, seed: int = 42, output_dir: Optional[Path] = None):
    splits, rejected, checksum = build_medmcqa(registry_path, seed)
    registry = load_registry(registry_path)
    out = output_dir or registry_path.resolve().parents[2] / registry["output"]["directory"]
    return write_splits(
        splits=splits, output_dir=out, dataset_id="medmcqa", seed=seed,
        source_revision=registry["source"]["revision"], source_checksum=checksum,
        source_checksum_method="sha256_of_revision_and_canonical_selected_records",
        rejected_row_counts=rejected,
        metadata={
            "allocation_counts": {key: len(value) for key, value in splits.items()},
            "test_source": "pinned MedMCQA validation split; used as pilot test data",
            "validation_source": "seeded disjoint sample from pinned MedMCQA train split",
        },
    )


def build_and_write_pubmedqa(registry_path: Path, seed: int = 42, output_dir: Optional[Path] = None):
    splits, rejected, checksum = build_pubmedqa(registry_path, seed)
    registry = load_registry(registry_path)
    out = output_dir or registry_path.resolve().parents[2] / registry["output"]["directory"]
    label_counts = {
        split_name: {label: sum(row["gold_candidate_id"] == label for row in rows) for label in ("yes", "no")}
        for split_name, rows in splits.items()
    }
    return write_splits(
        splits=splits, output_dir=out, dataset_id="pubmedqa", seed=seed,
        source_revision=registry["source"]["revision"], source_checksum=checksum,
        source_checksum_method="sha256_of_revision_and_canonical_selected_records",
        rejected_row_counts=rejected,
        metadata={
            "label_counts_by_split": label_counts,
            "label_provenance": "PQA-A automatically generated heuristic labels; validation and test are not expert-adjudicated",
            "label_balance": "50/50 yes/no in each split",
            "pqa_labeled_use": "reserved and excluded from this pilot",
        },
    )
