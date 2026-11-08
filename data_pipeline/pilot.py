from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .core import assert_group_disjoint, load_registry, validate_records, write_splits
from .hf_builders import build_medmcqa, build_pubmedqa
from .scale_builders import build_scale

ROOT = Path(__file__).resolve().parents[1]


def build_pilot(seed: int = 42) -> Dict[str, object]:
    pilot_registry = load_registry(ROOT / "data/registry/pilot.yaml")
    allocation = pilot_registry["allocation"]
    expected_splits = allocation["splits"]
    split_names = list(expected_splits)

    med_registry_path = ROOT / "data/registry/medmcqa.yaml"
    pub_registry_path = ROOT / "data/registry/pubmedqa.yaml"
    med_splits, med_rejected, med_checksum = build_medmcqa(med_registry_path, seed)
    pub_splits, pub_rejected, pub_checksum = build_pubmedqa(pub_registry_path, seed)

    med = load_registry(med_registry_path)
    pub = load_registry(pub_registry_path)
    write_splits(
        splits=med_splits,
        output_dir=ROOT / med["output"]["directory"],
        dataset_id="medmcqa",
        seed=seed,
        source_revision=med["source"]["revision"],
        source_checksum=med_checksum,
        source_checksum_method="sha256_of_revision_and_canonical_selected_records",
        rejected_row_counts=med_rejected,
        metadata={"test_source": "pinned MedMCQA validation split", "validation_source": "disjoint MedMCQA train rows"},
    )
    write_splits(
        splits=pub_splits,
        output_dir=ROOT / pub["output"]["directory"],
        dataset_id="pubmedqa",
        seed=seed,
        source_revision=pub["source"]["revision"],
        source_checksum=pub_checksum,
        source_checksum_method="sha256_of_revision_and_canonical_selected_records",
        rejected_row_counts=pub_rejected,
        metadata={
            "label_provenance": "PQA-A automatically generated heuristic labels; validation and test are not expert-adjudicated",
            "label_balance": "50/50 yes/no in each split",
            "pqa_labeled_use": "reserved and excluded from this pilot",
        },
    )

    score_splits: Dict[str, List[dict]] = {name: [] for name in split_names}
    score_counts_by_dataset: Dict[str, Dict[str, int]] = {}
    for dataset_id, configured_counts in allocation["score_by_dataset"].items():
        registry_path = ROOT / f"data/registry/{dataset_id}.yaml"
        counts = {name: int(configured_counts[name]) for name in split_names}
        scale_splits, rejected, checksum = build_scale(dataset_id, registry_path, seed, counts=counts)
        registry = load_registry(registry_path)
        write_splits(
            splits=scale_splits,
            output_dir=ROOT / registry["output"]["directory"],
            dataset_id=dataset_id,
            seed=seed,
            source_revision=registry["source"]["revision"],
            source_checksum=checksum,
            source_checksum_method="sha256_of_generator_revision_seed_and_canonical_records",
            rejected_row_counts=rejected,
            metadata={"synthetic_rule_labels": True, "pilot_allocation_counts": counts},
        )
        score_counts_by_dataset[dataset_id] = {name: len(rows) for name, rows in scale_splits.items()}
        for split_name in split_names:
            score_splits[split_name].extend(scale_splits[split_name])

    combined: Dict[str, List[dict]] = {}
    decision_type_counts: Dict[str, Dict[str, int]] = {}
    pubmed_label_counts: Dict[str, Dict[str, int]] = {}
    for split_name in split_names:
        combined[split_name] = med_splits[split_name] + pub_splits[split_name] + score_splits[split_name]
        expected_count = sum(int(value) for value in expected_splits[split_name].values())
        validate_records(combined[split_name], expected_count=expected_count)
        actual_types = {
            kind: sum(record["decision_type"] == kind for record in combined[split_name])
            for kind in ("choice", "noul", "score")
        }
        if actual_types != {kind: int(value) for kind, value in expected_splits[split_name].items()}:
            raise ValueError(f"{split_name} decision-type counts differ from allocation: {actual_types}")
        decision_type_counts[split_name] = actual_types
        label_counts = {
            label: sum(row["gold_candidate_id"] == label for row in pub_splits[split_name])
            for label in ("yes", "no")
        }
        expected_noul = int(expected_splits[split_name]["noul"]) // 2
        if label_counts != {"yes": expected_noul, "no": expected_noul}:
            raise ValueError(f"PQA-A {split_name} labels are not balanced: {label_counts}")
        pubmed_label_counts[split_name] = label_counts
    assert_group_disjoint(combined)

    manifest = write_splits(
        splits=combined,
        output_dir=ROOT / pilot_registry["output"]["directory"],
        dataset_id="pilot",
        seed=seed,
        source_revision=pilot_registry["source"]["revision"],
        source_checksum_method="sha256_of_canonical_final_records",
        rejected_row_counts={
            **{f"medmcqa_{key}": value for key, value in med_rejected.items()},
            **{f"pubmedqa_{key}": value for key, value in pub_rejected.items()},
        },
        metadata={
            "decision_type_counts_by_split": decision_type_counts,
            "pubmedqa_pqa_a_label_counts_by_split": pubmed_label_counts,
            "pubmedqa_label_provenance": "PQA-A automatically generated heuristic labels, including validation and test",
            "score_counts_by_dataset_and_split": score_counts_by_dataset,
            "mimic_esi_included": False,
            "mimic_esi_note": "Not included in this pilot build; awaiting a local source supplied by the user.",
        },
    )
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build the reproducible 20,000-record pilot and source datasets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_pilot(seed=args.seed)
    print(f"Built pilot: {manifest['split_counts']} (seed={manifest['seed']})")
    print(f"Details: {manifest['metadata']}")


if __name__ == "__main__":
    main()
