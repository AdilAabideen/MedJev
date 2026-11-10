# R02: decision requests and token encoding

R02 prepares input for the MedGemma decision model. It does not run the backbone or apply an attention mask. The mask, head, and training loop are later tickets.

## Request format

A request holds one shared state and named questions. A state can be text or JSON; objects and lists are rendered as readable text. The three question types use `question` for the instruction:

```python
from modeling.types import DecisionRequest

request = DecisionRequest.model_validate({
    "state": {"observation": "A patient has a fever."},
    "questions": {
        "fever": {"type": "noul", "question": "Does the patient have a fever?"},
        "scale": {
            "type": "score", "question": "What is the stated score?",
            "candidates": [{"id": "zero", "text": "0"}, {"id": "one", "text": "1"}],
            "ordered_values": [0, 1], "scale_id": "example",
        },
    },
})
```

Choice uses `type: "choice"` and the same `candidates` list as Score. Noul supplies fixed Yes/No candidates. Candidate IDs and Score metadata are for mapping outputs; only state, question, and candidate text enter the model. `DecisionRequest.model_validate_json(...)` accepts JSON text.

Use `parse_request(json_text)` to validate a JSON request. For a canonical Parquet row, `from_parquet_row(row)` returns `(request, targets)`. The target mapping is separate from the request; `target_indices(normalize_request(request), targets)` maps gold IDs to indices for a later training loop. Existing data and splits are untouched.

## Encoding

```python
from modeling.tokenizer.medgemma import load_tokenizer
from modeling.tokenizer.encode import DecisionEncoder

tokenizer = load_tokenizer()  # pinned Google checkpoint; gated access is required
encoder = DecisionEncoder(tokenizer)  # audits the five marker IDs
packed = encoder.encode(request)
print(encoder.marker_ids)
print(tokenizer.decode(packed["ids"], skip_special_tokens=False))
print(packed["spans"], packed["decide_idx"], packed["opt_idx"])
```

To inspect a real training row (with the data dependencies installed):

```python
import pyarrow.parquet as pq
from modeling.normalize import from_parquet_row, normalize_request, target_indices

row = pq.read_table("datasets/medmcqa/v1/processed/train.parquet").slice(0, 1).to_pylist()[0]
request, targets = from_parquet_row(row)
packed = encoder.encode(request)
print(target_indices(normalize_request(request), targets))
print(tokenizer.decode(packed["ids"], skip_special_tokens=False))
print(packed["spans"])
```

The chat template puts the following inside one Gemma user turn:

```text
<unused0> state
<unused1> question 1 <unused2> candidate <unused3> ... <unused4>
<unused1> question 2 <unused2> candidate <unused3> ... <unused4>
```

The marker names have roles assigned by MedJev: state, question, candidate start, candidate end, and decision. The pinned Google tokenizer audit returned IDs **6, 7, 8, 9, 10**, respectively. They must each be an existing distinct single token that decodes back to its spelling. The encoder stops if the audit fails. It escapes reserved marker and control-token spellings in user text to prevent forged boundaries.

`ids` are the model input. `seg` identifies shared context and question branches; `opt` identifies candidate spans and decision tokens; `pos` gives the option-isolated positions. `decide_idx` and `opt_idx` locate the decision and candidate-end tokens. Spans are half-open `[start, stop)`. Candidate and question IDs are returned as metadata, not tokenized. Position metadata alone does not isolate attention: R04 must apply and validate the mask inside MedGemma.

This adapts [Kev's encoder](https://github.com/jaredpalmer/kev/blob/main/kev/model.py) to MedGemma's tokenizer and chat template. To verify the actual gated checkpoint, run the example above on a machine with access to `google/medgemma-1.5-4b-it` at the pinned revision. No GPU or model weights are required for the tokenizer audit. Install `requirements-train.txt` to provide Transformers, Jinja2, and Pydantic.
