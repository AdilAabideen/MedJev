"""Fast deterministic checks using in-memory fixtures; no dataset download or PHI required."""
from __future__ import annotations

import csv
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from .core import assert_group_disjoint, load_registry, validate_records
from . import hf_builders
from .mimic_builder import build_mimic_esi
from .scale_builders import build_scale, curb65_score, news2_score

ROOT = Path(__file__).resolve().parents[1]


class FakeDataset(list):
    def shuffle(self, seed: int):
        values = list(self)
        random.Random(seed).shuffle(values)
        return FakeDataset(values)


def _compare_splits(left, right) -> None:
    assert list(left) == list(right)
    for split in left:
        assert [(row["example_id"], row["gold_candidate_id"]) for row in left[split]] == [
            (row["example_id"], row["gold_candidate_id"]) for row in right[split]
        ]
        validate_records(left[split])
    assert_group_disjoint(left)


def _check_medmcqa() -> None:
    registry = load_registry(ROOT / "data/registry/medmcqa.yaml")
    registry["source"]["revision"] = "fixture-revision"
    registry["source_splits"]["train"]["expected_rows"] = 8
    registry["source_splits"]["validation"]["expected_rows"] = 2
    registry["internal_allocation"]["counts"] = {
        "train": 4, "validation": 2
    }
    registry["internal_allocation"]["external_test_count"] = 2
    rows = FakeDataset([
        {"id": i, "question": f"Question {i}?", "opa": "Alpha", "opb": "Beta", "opc": "Gamma", "opd": "Delta",
         "cop": i % 4, "subject_name": "fixture", "topic_name": "test", "choice_type": "single"}
        for i in range(8)
    ])
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "medmcqa.yaml"
        path.write_text(yaml.safe_dump(registry), encoding="utf-8")
        validation_rows = FakeDataset([
            {"id": 100 + i, "question": f"Validation {i}?", "opa": "Alpha", "opb": "Beta", "opc": "Gamma", "opd": "Delta",
             "cop": i % 4, "subject_name": "fixture", "topic_name": "validation", "choice_type": "single"}
            for i in range(2)
        ])
        def fake_med_load(_repo, _config, _revision, split="train"):
            return rows if split == "train" else validation_rows
        with patch.object(hf_builders, "_load_dataset", side_effect=fake_med_load):
            first, _, _ = hf_builders.build_medmcqa(path, seed=42)
            second, _, _ = hf_builders.build_medmcqa(path, seed=42)
    assert {key: len(value) for key, value in first.items()} == {
        "train": 4, "validation": 2, "test": 2
    }
    _compare_splits(first, second)


def _pubmed_row(pubid: int, label: str) -> dict:
    return {
        "pubid": pubid,
        "question": f"Does intervention {pubid} help?",
        "context": {"contexts": [f"Study context for article {pubid}."]},
        "long_answer": "excluded long answer",
        "final_decision": label,
    }


def _check_pubmedqa() -> None:
    registry = load_registry(ROOT / "data/registry/pubmedqa.yaml")
    registry["source"]["revision"] = "fixture-revision"
    registry["training_allocation"]["split_counts"] = {
        "train": {"yes": 1, "no": 1},
        "validation": {"yes": 1, "no": 1},
        "test": {"yes": 0, "no": 0},
    }
    registry["source_splits"]["pqa_artificial"]["expected_rows"] = 6
    registry["source_splits"]["pqa_labeled"]["expected_rows"] = 25
    artificial = FakeDataset([
        _pubmed_row(1, "yes"), _pubmed_row(2, "yes"), _pubmed_row(3, "yes"),
        _pubmed_row(4, "no"), _pubmed_row(5, "no"), _pubmed_row(6, "no"),
    ])
    labeled = FakeDataset(
        [_pubmed_row(4, "no")]
        + [_pubmed_row(100 + i, "yes") for i in range(10)]
        + [_pubmed_row(200 + i, "no") for i in range(10)]
        + [_pubmed_row(300 + i, "maybe") for i in range(4)]
    )
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "pubmedqa.yaml"
        path.write_text(yaml.safe_dump(registry), encoding="utf-8")
        def fake_load(_repo, config, _revision):
            return artificial if config == "pqa_artificial" else labeled
        with patch.object(hf_builders, "_load_dataset", side_effect=fake_load):
            first, rejected, _ = hf_builders.build_pubmedqa(path, seed=42)
            second, _, _ = hf_builders.build_pubmedqa(path, seed=42)
    assert len(first["train"]) == 2
    assert {label: sum(row["gold_candidate_id"] == label for row in first["train"]) for label in ("yes", "no")} == {"yes": 1, "no": 1}
    assert rejected["cross_split_article_overlap_excluded"] >= 1
    _compare_splits(first, second)
    assert_group_disjoint({key: rows for key, rows in first.items()})
    assert all("excluded long answer" not in row["state"] for rows in first.values() for row in rows)


def _check_scales() -> None:
    for dataset_id in ("news2", "gcs", "curb65"):
        registry_path = ROOT / f"data/registry/{dataset_id}.yaml"
        counts = {"train": 3, "validation": 2, "test": 1}
        first, _, first_checksum = build_scale(dataset_id, registry_path, seed=42, counts=counts)
        second, _, second_checksum = build_scale(dataset_id, registry_path, seed=42, counts=counts)
        _compare_splits(first, second)
        assert first_checksum == second_checksum
        assert {name: len(rows) for name, rows in first.items()} == counts
        assert all(row["gold_candidate_id"] in {candidate["id"] for candidate in row["candidates"]} for row in first["train"])
    assert news2_score(12, 96, 120, 70, 37.0, False, False) == 0
    assert news2_score(8, 90, 90, 40, 35.0, True, True) == 20
    assert curb65_score(65, True, 7.1, 30, 89, 60) == 5


def _check_mimic_converter() -> None:
    registry_path = ROOT / "data/registry/mimic_esi.yaml"
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "triage.csv"
        rows = [
            {"subject_id": "patient-secret-1", "stay_id": "20", "chiefcomplaint": "abdominal pain", "heartrate": 80, "resprate": 16, "o2sat": 98, "sbp": 120, "dbp": 80, "pain": 3, "temperature": 37, "acuity": 3},
            {"subject_id": "patient-secret-1", "stay_id": "10", "chiefcomplaint": "chest discomfort", "heartrate": 90, "resprate": 18, "o2sat": 96, "sbp": 110, "dbp": 70, "pain": 4, "temperature": 36, "acuity": 2},
            {"subject_id": "patient-secret-2", "stay_id": "30", "chiefcomplaint": "headache", "heartrate": 72, "resprate": 14, "o2sat": 99, "sbp": 130, "dbp": 82, "pain": 2, "temperature": 37, "acuity": 4},
        ]
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        first, _, checksum1 = build_mimic_esi(registry_path, source, seed=42, count=200)
        second, _, checksum2 = build_mimic_esi(registry_path, source, seed=42, count=200)
    _compare_splits(first, second)
    assert checksum1 == checksum2
    assert len(first["train"]) == 2  # One stay per patient.
    assert all("patient-secret" not in row["example_id"] + row["source_group_id"] + row["state"] for row in first["train"])
    assert all("acuity" not in row["state"].lower() and "temperature" not in row["state"].lower() for row in first["train"])


def main() -> None:
    _check_medmcqa()
    _check_pubmedqa()
    _check_scales()
    _check_mimic_converter()
    print("Self-check passed: deterministic IDs/labels, exact fixture allocations, valid targets, and patient-local ESI handling.")


if __name__ == "__main__":
    main()
