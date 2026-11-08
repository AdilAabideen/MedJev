from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .core import load_registry, make_score_record, records_checksum, split_records_exact, write_splits


def _news2_component_scores(rr: int, spo2: int, sbp: int, pulse: int, temperature: float, confusion: bool, oxygen: bool) -> Dict[str, int]:
    if rr <= 8:
        rr_score = 3
    elif rr <= 11:
        rr_score = 1
    elif rr <= 20:
        rr_score = 0
    elif rr <= 24:
        rr_score = 2
    else:
        rr_score = 3

    spo2_score = 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0

    if sbp <= 90:
        sbp_score = 3
    elif sbp <= 100:
        sbp_score = 2
    elif sbp <= 110:
        sbp_score = 1
    elif sbp <= 219:
        sbp_score = 0
    else:
        sbp_score = 3

    if pulse <= 40:
        pulse_score = 3
    elif pulse <= 50:
        pulse_score = 1
    elif pulse <= 90:
        pulse_score = 0
    elif pulse <= 110:
        pulse_score = 1
    elif pulse <= 130:
        pulse_score = 2
    else:
        pulse_score = 3

    if temperature <= 35.0:
        temperature_score = 3
    elif temperature <= 36.0:
        temperature_score = 1
    elif temperature <= 38.0:
        temperature_score = 0
    elif temperature <= 39.0:
        temperature_score = 1
    else:
        temperature_score = 2

    return {
        "respiratory_rate": rr_score,
        "oxygen_saturation": spo2_score,
        "systolic_blood_pressure": sbp_score,
        "pulse": pulse_score,
        "temperature": temperature_score,
        "consciousness": 3 if confusion else 0,
        "supplemental_oxygen": 2 if oxygen else 0,
    }


def news2_score(rr: int, spo2: int, sbp: int, pulse: int, temperature: float, confusion: bool, oxygen: bool) -> int:
    return sum(_news2_component_scores(rr, spo2, sbp, pulse, temperature, confusion, oxygen).values())


def curb65_score(age: int, confusion: bool, urea: float, rr: int, sbp: int, dbp: int) -> int:
    return int(confusion) + int(urea > 7.0) + int(rr >= 30) + int(sbp < 90 or dbp <= 60) + int(age >= 65)


_EYE = {1: "no eye opening", 2: "eye opening to pressure", 3: "eye opening to sound", 4: "eyes open spontaneously"}
_VERBAL = {1: "no verbal response", 2: "incomprehensible sounds", 3: "inappropriate words", 4: "confused conversation", 5: "oriented conversation"}
_MOTOR = {1: "no motor response", 2: "extension to pressure", 3: "abnormal flexion to pressure", 4: "withdrawal from pressure", 5: "localizes pressure", 6: "obeys commands"}


def _news2_examples(count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    values = list(range(0, 21))
    for i in range(count):
        rr = rng.randint(6, 40)
        spo2 = rng.randint(75, 100)
        sbp = rng.randint(65, 240)
        pulse = rng.randint(32, 160)
        temperature = round(rng.uniform(33.0, 42.0), 1)
        confusion = rng.random() < 0.12
        oxygen = rng.random() < 0.25
        score = news2_score(rr, spo2, sbp, pulse, temperature, confusion, oxygen)
        state = (
            f"Respiratory rate: {rr}/min; SpO2: {spo2}% (Scale 1); systolic blood pressure: {sbp} mmHg; "
            f"pulse: {pulse}/min; temperature: {temperature:.1f} C; new confusion/CVPU: {'yes' if confusion else 'no'}; "
            f"supplemental oxygen: {'yes' if oxygen else 'no'}."
        )
        records.append(make_score_record(
            dataset_id="news2",
            raw_source_id=f"synthetic:{seed}:{i}",
            state=state,
            question="What is the NEWS2 aggregate score for this adult observation set using SpO2 Scale 1?",
            score=score,
            scale_id="news2_scale1",
            ordered_values=values,
            score_direction="higher_is_worse",
        ))
    return records


def _gcs_examples(count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    values = list(range(3, 16))
    for i in range(count):
        eye, verbal, motor = rng.randint(1, 4), rng.randint(1, 5), rng.randint(1, 6)
        score = eye + verbal + motor
        state = (
            f"Eye response: E{eye} ({_EYE[eye]}); verbal response: V{verbal} ({_VERBAL[verbal]}); "
            f"motor response: M{motor} ({_MOTOR[motor]})."
        )
        records.append(make_score_record(
            dataset_id="gcs",
            raw_source_id=f"synthetic:{seed}:{i}",
            state=state,
            question="What is the total Glasgow Coma Scale score from these eye, verbal, and motor responses?",
            score=score,
            scale_id="glasgow_coma_scale",
            ordered_values=values,
            score_direction="higher_is_better",
        ))
    return records


def _curb65_examples(count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    values = list(range(6))
    for i in range(count):
        age = rng.randint(18, 100)
        confusion = rng.random() < 0.15
        urea = round(rng.uniform(1.0, 25.0), 1)
        rr = rng.randint(8, 45)
        sbp = rng.randint(65, 210)
        dbp = rng.randint(35, 120)
        score = curb65_score(age, confusion, urea, rr, sbp, dbp)
        state = (
            f"Age: {age} years; new confusion: {'yes' if confusion else 'no'}; serum urea: {urea:.1f} mmol/L; "
            f"respiratory rate: {rr}/min; systolic blood pressure: {sbp} mmHg; diastolic blood pressure: {dbp} mmHg."
        )
        records.append(make_score_record(
            dataset_id="curb65",
            raw_source_id=f"synthetic:{seed}:{i}",
            state=state,
            question="What is the CURB-65 score from the stated findings?",
            score=score,
            scale_id="curb65",
            ordered_values=values,
            score_direction="higher_is_worse",
        ))
    return records


_GENERATORS = {"news2": _news2_examples, "gcs": _gcs_examples, "curb65": _curb65_examples}


def build_scale(
    dataset_id: str,
    registry_path: Path,
    seed: int = 42,
    count: Optional[int] = None,
    counts: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], str]:
    if dataset_id not in _GENERATORS:
        raise ValueError(f"No synthetic scale builder for {dataset_id}")
    registry = load_registry(registry_path)
    if counts is not None and count is not None:
        raise ValueError("Pass either an exact split allocation or a train count override, not both")
    if counts is None:
        counts = {name: int(value) for name, value in registry["training_allocation"]["counts"].items()}
        if count is not None:
            counts["train"] = count
    if any(value < 0 for value in counts.values()):
        raise ValueError("Split counts cannot be negative")
    requested = sum(counts.values())
    records = _GENERATORS[dataset_id](requested, seed)
    splits = split_records_exact(records, counts, seed)
    checksum = records_checksum(records, registry["source"]["revision"] + f":seed={seed}")
    return splits, {}, checksum


def build_and_write_scale(
    dataset_id: str,
    registry_path: Path,
    seed: int = 42,
    count: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    splits, rejected, checksum = build_scale(dataset_id, registry_path, seed, count=count)
    registry = load_registry(registry_path)
    out = output_dir or registry_path.resolve().parents[2] / registry["output"]["directory"]
    return write_splits(
        splits=splits, output_dir=out, dataset_id=dataset_id, seed=seed,
        source_revision=registry["source"]["revision"], source_checksum=checksum,
        source_checksum_method="sha256_of_generator_revision_seed_and_canonical_records",
        rejected_row_counts=rejected,
        metadata={
            "rule_reference": registry["source"].get("rules_reference"),
            "synthetic_rule_labels": True,
            "allocation_counts": {name: len(rows) for name, rows in splits.items()},
        },
    )
