"""Load the pinned tokenizer and audit the reserved marker IDs."""

MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
MARKERS = {
    "state": "<unused0>",
    "question": "<unused1>",
    "candidate_start": "<unused2>",
    "candidate_end": "<unused3>",
    "decide": "<unused4>",
}
EXPECTED_MARKER_IDS = dict(zip(MARKERS, (6, 7, 8, 9, 10)))


def load_tokenizer():
    """Load the same tokenizer revision as R01; model weights are not loaded."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def audit_markers(tokenizer) -> dict[str, int]:
    """Require each marker to be a distinct, reversible, existing single token."""
    marker_ids = {}
    for name, marker in MARKERS.items():
        token_id = tokenizer.convert_tokens_to_ids(marker)
        encoded = tokenizer(marker, add_special_tokens=False).input_ids
        if (
            not isinstance(token_id, int)
            or token_id != EXPECTED_MARKER_IDS[name]
            or token_id == tokenizer.unk_token_id
            or encoded != [token_id]
            or tokenizer.decode([token_id], skip_special_tokens=False) != marker
        ):
            raise ValueError(f"Tokenizer marker audit failed: {marker}")
        marker_ids[name] = token_id
    if len(set(marker_ids.values())) != len(MARKERS):
        raise ValueError("Tokenizer marker IDs must be distinct")
    return marker_ids
