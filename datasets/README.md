# MedJev pilot dataset builders

The builders create canonical Parquet records plus a JSON manifest. The versioned Python scripts are kept beside each registry entry. Raw downloads, generated records, and manifests are git-ignored.

## Setup

From the repository root, install the pinned data-builder dependencies:

```bash
python -m pip install -r requirements-data.txt
```


## Canonical record shape

Every Parquet row has `example_id`, `source_group_id`, `dataset_id`, `decision_type`, `state`, `question`, `candidates`, and `gold_candidate_id`. `example_id` and `source_group_id` are stable SHA-256 identifiers; raw source IDs are not written. Each candidate has a stable `id`, human-readable `text`, and `order`. Score rows also have `scale_id`, `ordered_values`, and `score_direction`; Choice and Noul rows use null/empty score metadata.

For example, a GCS row is shaped like this (IDs abbreviated):

```json
{
  "example_id": "<sha256>",
  "source_group_id": "<sha256>",
  "dataset_id": "gcs",
  "decision_type": "score",
  "state": "Eye response: E3 (eyes open to sound); verbal response: V4 (confused conversation); motor response: M6 (obeys commands).",
  "question": "What is the total Glasgow Coma Scale score from these eye, verbal, and motor responses?",
  "candidates": [{"id": "value_3", "text": "3", "order": 0}, {"id": "value_4", "text": "4", "order": 1}, {"id": "value_5", "text": "5", "order": 2}, {"id": "value_6", "text": "6", "order": 3}, {"id": "value_7", "text": "7", "order": 4}, {"id": "value_8", "text": "8", "order": 5}, {"id": "value_9", "text": "9", "order": 6}, {"id": "value_10", "text": "10", "order": 7}, {"id": "value_11", "text": "11", "order": 8}, {"id": "value_12", "text": "12", "order": 9}, {"id": "value_13", "text": "13", "order": 10}, {"id": "value_14", "text": "14", "order": 11}, {"id": "value_15", "text": "15", "order": 12}],
  "gold_candidate_id": "value_13",
  "scale_id": "glasgow_coma_scale",
  "ordered_values": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "score_direction": "higher_is_better"
}
```

`gold_candidate_id` points to one of the candidate IDs.

## Build individual sources

```bash
python datasets/medmcqa/v1/build.py --seed 42
python datasets/pubmedqa/v1/build.py --seed 42
python datasets/news2/v1/build.py --seed 42
python datasets/gcs/v1/build.py --seed 42
python datasets/curb65/v1/build.py --seed 42
```

MedMCQA writes 5,000 training examples and 2,500 validation examples sampled without replacement from its pinned training split. Its pinned validation split supplies 2,500 test examples. PubMedQA uses PQA-A for all three splits: 3,000 training, 1,500 validation, and 1,500 test records. Every PubMedQA split is balanced 50/50 between `yes` and `no`. PQA-A labels are automatically generated heuristic labels, so the validation and test scores are not expert-adjudicated; this provenance is recorded in each manifest. PQA-L remains reserved and is not part of this pilot.

NEWS2, GCS, and CURB-65 each write `train.parquet`, `validation.parquet`, and `test.parquet`. The cases are seeded generated examples whose score labels are computed directly from the cited scoring rules. They teach rule arithmetic and output mapping; they are not independently labelled clinical judgments.

## Local MIMIC ESI training source and combined pilot

The MIMIC builder reads the local MIMIC-IV-ED export in place. It does not download or copy the source file. The supplied MimicDataset.csv is stored under datasets/mimic_esi/v1/static/ and is git-ignored.

    python datasets/mimic_esi/v1/build.py --input datasets/mimic_esi/v1/static/MimicDataset.csv --seed 42
    python datasets/build_pilot.py --seed 42

The MIMIC source builder writes 200 training records only. Its stratified sample includes at least five examples from each nurse-assigned acuity level, then allocates the remaining examples according to the clean source distribution. The supplied export has stay_id but not subject_id, so stay_id is used to identify visits; patient-level separation cannot be guaranteed. Raw visit IDs are hashed in record identifiers and omitted from model state and manifests. The combined pilot continues to exclude MIMIC until patient-disjoint MIMIC evaluation splits are available.

The combined pilot writes 10,000 train, 5,000 validation, and 5,000 test records. Each split retains the same 50% Choice, 30% Noul, and 20% Score mix. Score examples in the combined pilot remain split across NEWS2, GCS, and CURB-65.

## External evaluation registries

`data/registry/pedalfast_gcs.yaml` and `data/registry/nacef_curb65.yaml` are evaluation-only. PEDALFAST is pediatric TBI data and must be reported separately. NACeF requires credentialed PhysioNet access. Neither source is included in the training build or automatically downloaded.

## Fast repeatability checks

```bash
python -m data_pipeline.selfcheck
```

This uses in-memory fixtures and generated cases. It does not download the source datasets or read any real MIMIC data.
