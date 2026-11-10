"""Convert typed requests and canonical dataset rows into model inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import Choice, DecisionRequest, Noul, Score


def parse_request(value: str | bytes | Mapping[str, Any]) -> DecisionRequest:
    """Validate JSON text or a decoded request object."""
    if isinstance(value, (str, bytes)):
        return DecisionRequest.model_validate_json(value)
    return DecisionRequest.model_validate(value)


def render(value: Any, indent: int = 0) -> str:
    """Render JSON state as readable text, preserving the supplied field order."""
    if value is None:
        return ""
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            prefix = "  " * indent + str(key) + ":"
            if isinstance(item, (dict, list)):
                lines.append(prefix + "\n" + render(item, indent + 1))
            else:
                lines.append(prefix + " " + render(item, indent + 1))
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join("  " * indent + "- " + render(item, indent + 1).lstrip() for item in value)
    return str(value)


def normalize_request(request: DecisionRequest) -> dict[str, Any]:
    """Return the shared state and question branches used by the encoder."""
    questions = []
    for question_id, question in request.questions.items():
        if isinstance(question, Noul):
            candidates = [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
        else:
            candidates = [candidate.model_dump() for candidate in question.candidates]
        item = {
            "id": question_id,
            "type": question.type,
            "question": question.question,
            "candidates": candidates,
        }
        if isinstance(question, Score):
            item.update(
                ordered_values=list(question.ordered_values),
                scale_id=question.scale_id,
                score_direction=question.score_direction,
            )
        questions.append(item)
    return {"state": render(request.state), "questions": questions}


def from_parquet_row(row: Mapping[str, Any]) -> tuple[DecisionRequest, dict[str, str]]:
    """Convert one canonical row to an unlabeled request and separate gold ID."""
    kind = row["decision_type"]
    source_candidates = row["candidates"]
    candidates = [{"id": item["id"], "text": item["text"]} for item in source_candidates]
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)) or row["gold_candidate_id"] not in ids:
        raise ValueError("invalid candidate IDs or gold candidate ID")

    if kind == "noul":
        labels = {item["id"]: item["text"].casefold() for item in candidates}
        if labels != {"yes": "yes", "no": "no"}:
            raise ValueError("Noul rows must have Yes and No candidates")
        question = Noul(type="noul", question=row["question"])
    elif kind == "choice":
        question = Choice(type="choice", question=row["question"], candidates=candidates)
    elif kind == "score":
        if [str(value) for value in row["ordered_values"]] != [item["text"] for item in candidates]:
            raise ValueError("Score candidate texts must match ordered_values")
        question = Score(
            type="score", question=row["question"], candidates=candidates,
            ordered_values=row["ordered_values"], scale_id=row.get("scale_id"),
            score_direction=row.get("score_direction"),
        )
    else:
        raise ValueError(f"unknown decision type: {kind!r}")
    request = DecisionRequest(state=row["state"], questions={"decision": question})
    return request, {"decision": row["gold_candidate_id"]}


def target_indices(record: Mapping[str, Any], targets: Mapping[str, str]) -> dict[str, int]:
    """Map separate gold IDs to candidate indices without changing model inputs."""
    result = {}
    for question in record["questions"]:
        question_id = question["id"]
        ids = [candidate["id"] for candidate in question["candidates"]]
        try:
            result[question_id] = ids.index(targets[question_id])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"missing or invalid target for {question_id!r}") from exc
    if set(result) != set(targets):
        raise ValueError("targets contain unknown question IDs")
    return result
