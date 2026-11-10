#!/usr/bin/env python3
"""Show how one Choice, Noul, and Score request becomes encoder input."""

import argparse
import json
from pathlib import Path
from pprint import pprint
import sys

# Allow `python scripts/inspect_tokenizer.py` from a repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.normalize import normalize_request, parse_request
from modeling.tokenizer.encode import DecisionEncoder
from modeling.tokenizer.medgemma import load_tokenizer


EXAMPLES = {
    "Choice": {
        "state": "A patient has a temperature of 38.2 C and a pulse of 90/min.",
        "questions": {
            "vital_sign": {
                "type": "choice",
                "question": "Which option is the patient's pulse?",
                "candidates": [
                    {"id": "temperature", "text": "38.2 C"},
                    {"id": "pulse", "text": "90/min"},
                ],
            }
        },
    },
    "Noul": {
        "state": "A patient's temperature is 38.2 C.",
        "questions": {
            "fever": {"type": "noul", "question": "Is the temperature at least 38 C?"}
        },
    },
    "Score": {
        "state": (
            "Age: 45 years; confusion: no; urea: 5 mmol/L; "
            "respiratory rate: 20/min; blood pressure: 130/80 mmHg."
        ),
        "questions": {
            "curb65": {
                "type": "score",
                "question": "What is the CURB-65 score?",
                "candidates": [{"id": f"value_{value}", "text": str(value)} for value in range(6)],
                "ordered_values": list(range(6)),
                "scale_id": "curb65",
                "score_direction": "higher_is_worse",
            }
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-path",
        help="Local folder containing the pinned MedGemma tokenizer files",
    )
    args = parser.parse_args()

    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    else:
        tokenizer = load_tokenizer()
    encoder = DecisionEncoder(tokenizer)
    print("Marker audit:", encoder.marker_ids)

    for name, example in EXAMPLES.items():
        json_input = json.dumps(example, indent=2, ensure_ascii=False)
        request = parse_request(json_input)
        normalized = normalize_request(request)
        packed = encoder.encode(request)

        print(f"\n{'=' * 24} {name} {'=' * 24}")
        print("\n1. JSON input:")
        print(json_input)
        print("\n2. Normalized record used inside the encoder:")
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
        print("\n3. Decoded packed text (the tokens shown as text):")
        print(tokenizer.decode(packed["ids"], skip_special_tokens=False))
        print("\n4. Encoder output (token IDs and position metadata):")
        pprint(packed, width=100, sort_dicts=False, compact=True)


if __name__ == "__main__":
    main()
