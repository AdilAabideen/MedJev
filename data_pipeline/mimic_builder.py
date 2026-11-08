from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .core import load_registry, make_score_record, sha256_file, write_splits

_REQUIRED = ("stay_id", "chiefcomplaint", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity")
_ACUITY_LEVELS = (1, 2, 3, 4, 5)


def _number(row: Mapping[str, Any], field: str, *, integer: bool = False) -> Optional[float]:
    raw = row.get(field)
    if raw is None or str(raw).strip() in {"", "\\N", "nan", "NaN", "None"}:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if not (number == number and abs(number) != float("inf")):
        return None
    if integer and not number.is_integer():
        return None
    return number


def _stratified_quotas(counts: Mapping[int, int], requested: int) -> Dict[int, int]:
    """Keep every acuity level represented, then allocate the remainder by source prevalence."""
    if requested < len(_ACUITY_LEVELS):
        raise ValueError("MIMIC ESI sample must contain at least one row for each acuity level")
    if any(counts.get(level, 0) < 1 for level in _ACUITY_LEVELS):
        missing = [level for level in _ACUITY_LEVELS if counts.get(level, 0) < 1]
        raise ValueError(f"Cannot sample all ESI levels; source has no clean rows for levels {missing}")

    minimum = min(5, requested // len(_ACUITY_LEVELS))
    quotas = {level: minimum for level in _ACUITY_LEVELS}
    remaining = requested - sum(quotas.values())
    total = sum(counts.values())
    exact = {level: remaining * counts[level] / total for level in _ACUITY_LEVELS}
    floors = {level: math.floor(exact[level]) for level in _ACUITY_LEVELS}
    for level in _ACUITY_LEVELS:
        quotas[level] += floors[level]
    leftover = remaining - sum(floors.values())
    priority = sorted(_ACUITY_LEVELS, key=lambda level: (-(exact[level] - floors[level]), level))
    for level in priority[:leftover]:
        quotas[level] += 1
    for level in _ACUITY_LEVELS:
        if quotas[level] > counts[level]:
            raise ValueError(
                f"Requested {quotas[level]} clean rows for ESI {level}, "
                f"but only {counts[level]} are available"
            )
    return quotas


def build_mimic_esi(registry_path: Path, input_path: Optional[Path] = None, seed: int = 42, count: int = 200):
    """Read a local MIMIC-IV-ED triage export; stay_id identifies each training visit."""
    registry = load_registry(registry_path)
    if input_path is None:
        env_name = registry["source"].get("local_path_env", "MIMIC_IV_ED_TRIAGE_CSV")
        raw_path = os.environ.get(env_name)
        if not raw_path:
            raise ValueError(f"Provide --input or set {env_name} to your local triage CSV path")
        input_path = Path(raw_path).expanduser()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"MIMIC-IV-ED triage CSV not found: {input_path}")
    if count < 0:
        raise ValueError("Count cannot be negative")

    valid_by_visit: Dict[str, Dict[str, Any]] = {}
    rejected = {"missing_required_column": 0, "missing_or_invalid_triage_fields": 0, "duplicate_stay_discarded": 0}
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        missing_columns = set(_REQUIRED) - header
        if missing_columns:
            rejected["missing_required_column"] = 1
            raise ValueError(f"MIMIC triage table is missing required columns: {sorted(missing_columns)}")
        for row in reader:
            visit_id = str(row.get("stay_id", "")).strip()
            complaint = " ".join(str(row.get("chiefcomplaint", "")).split())
            if complaint.lower() in {"\\N", "nan", "none", "null"}:
                complaint = ""
            vitals = {
                field: _number(row, field, integer=(field != "o2sat"))
                for field in ("heartrate", "resprate", "o2sat", "sbp", "dbp")
            }
            acuity = _number(row, "acuity", integer=True)
            if not visit_id or visit_id == "\\N" or not complaint or any(value is None for value in vitals.values()) or acuity not in _ACUITY_LEVELS:
                rejected["missing_or_invalid_triage_fields"] += 1
                continue
            if not (
                20 <= vitals["heartrate"] <= 300
                and 1 <= vitals["resprate"] <= 100
                and 1 <= vitals["o2sat"] <= 100
                and 20 <= vitals["sbp"] <= 350
                and 10 <= vitals["dbp"] <= 250
            ):
                rejected["missing_or_invalid_triage_fields"] += 1
                continue

            candidate = {
                "visit_id": visit_id,
                "complaint": complaint,
                "vitals": vitals,
                "acuity": int(acuity),
                "pain": _number(row, "pain", integer=True),
            }
            if visit_id in valid_by_visit:
                rejected["duplicate_stay_discarded"] += 1
                continue
            valid_by_visit[visit_id] = candidate

    rows_by_acuity: Dict[int, List[Dict[str, Any]]] = {level: [] for level in _ACUITY_LEVELS}
    for item in valid_by_visit.values():
        rows_by_acuity[item["acuity"]].append(item)
    available = {level: len(rows) for level, rows in rows_by_acuity.items()}
    if count > sum(available.values()):
        raise ValueError(f"Requested {count} clean visits, found {sum(available.values())}")
    quotas = _stratified_quotas(available, count)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for level in _ACUITY_LEVELS:
        candidates = rows_by_acuity[level][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:quotas[level]])
    rng.shuffle(selected)

    values = list(_ACUITY_LEVELS)
    records: List[Dict[str, Any]] = []
    for item in selected:
        v = item["vitals"]
        state = (
            f"Chief complaint: {item['complaint']}\n"
            f"Triage measurements: heart rate {v['heartrate']:g}/min; respiratory rate {v['resprate']:g}/min; "
            f"oxygen saturation {v['o2sat']:g}%; systolic blood pressure {v['sbp']:g} mmHg; "
            f"diastolic blood pressure {v['dbp']:g} mmHg."
        )
        if item["pain"] is not None and 0 <= item["pain"] <= 10:
            state += f" Pain score: {int(item['pain'])}/10."
        records.append(make_score_record(
            dataset_id="mimic_esi",
            raw_source_id=item["visit_id"],
            state=state,
            question="Which Emergency Severity Index acuity level was assigned at triage?",
            score=item["acuity"],
            scale_id="emergency_severity_index",
            ordered_values=values,
            score_direction="lower_is_more_urgent",
        ))

    return {"train": records}, rejected, sha256_file(input_path), quotas


def build_and_write_mimic_esi(registry_path: Path, input_path: Optional[Path] = None, seed: int = 42, count: int = 200, output_dir: Optional[Path] = None):
    splits, rejected, checksum, quotas = build_mimic_esi(registry_path, input_path, seed, count)
    registry = load_registry(registry_path)
    out = output_dir or registry_path.resolve().parents[2] / registry["output"]["directory"]
    return write_splits(
        splits=splits,
        output_dir=out,
        dataset_id="mimic_esi",
        seed=seed,
        source_revision=registry["source"]["revision"],
        source_checksum=checksum,
        source_checksum_method="sha256_of_local_input_csv; input path and raw IDs are not written to manifest",
        rejected_row_counts=rejected,
        metadata={
            "requested_train": count,
            "actual_train": len(splits["train"]),
            "unique_stay_groups": len(splits["train"]),
            "grouping_unit": "stay_id",
            "patient_disjointness_guaranteed": False,
            "acuity_excluded_from_state": True,
            "temperature_excluded": True,
            "sampled_acuity_counts": {str(level): quotas[level] for level in _ACUITY_LEVELS},
            "sampling_policy": "minimum of five examples per acuity level, remaining quota allocated by clean-source prevalence",
        },
    )
