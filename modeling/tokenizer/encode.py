"""Pack a decision request into token IDs and Kev-style branch metadata."""

from __future__ import annotations

from typing import Any

from modeling.normalize import normalize_request
from modeling.types import DecisionRequest
from .medgemma import MARKERS, audit_markers

OPT_NONE = -1
OPT_DECIDE = -2


class DecisionEncoder:
    """Reuse one audited tokenizer for many requests; no model or GPU is needed."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.marker_ids = audit_markers(tokenizer)
        # Fast tokenizers may recognize literal special tokens even with
        # add_special_tokens=False. Escape their spellings in supplied text.
        self._reserved = set(MARKERS.values()) | {
            token for token in tokenizer.all_special_tokens
            if token.startswith("<") and token.endswith(">")
        }

    def _text(self, text: str) -> str:
        for token in sorted(self._reserved, key=len, reverse=True):
            text = text.replace(token, "‹" + token[1:])
        return text

    def encode(self, request: DecisionRequest) -> dict[str, Any]:
        record = normalize_request(request)
        questions = record["questions"]
        body = [MARKERS["state"], self._text(record["state"])]
        expected = [self.marker_ids["state"]]
        for question in questions:
            body += [MARKERS["question"], self._text(question["question"])]
            expected.append(self.marker_ids["question"])
            for candidate in question["candidates"]:
                body += [MARKERS["candidate_start"], self._text(candidate["text"]), MARKERS["candidate_end"]]
                expected += [self.marker_ids["candidate_start"], self.marker_ids["candidate_end"]]
            body.append(MARKERS["decide"])
            expected.append(self.marker_ids["decide"])

        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "".join(body)}],
            tokenize=True, add_generation_prompt=False, return_dict=False, truncation=False,
        )
        ids = list(ids)
        marker_set = set(self.marker_ids.values())
        markers = [(index, token_id) for index, token_id in enumerate(ids) if token_id in marker_set]
        if [token_id for _, token_id in markers] != expected:
            raise ValueError("Packed marker positions do not match the request")
        bos = self.tokenizer.bos_token_id
        if bos is None or not ids or ids[0] != bos or ids.count(bos) != 1:
            raise ValueError("Expected one BOS at the start of the Gemma chat template")

        state_idx = markers[0][0]
        first_question = markers[1][0]
        seg = [0] * len(ids)
        pos = list(range(len(ids)))
        opt = [OPT_NONE] * len(ids)
        decisions, endpoints, spans, metadata = [], [], [], []
        cursor = 1
        for question_number, question in enumerate(questions, start=1):
            question_idx = markers[cursor][0]
            cursor += 1
            candidate_spans = []
            for candidate_number, candidate in enumerate(question["candidates"]):
                start, end = markers[cursor][0], markers[cursor + 1][0]
                cursor += 2
                candidate_spans.append({
                    "candidate_id": candidate["id"],
                    "span": (start, end + 1),
                    "text_span": (start + 1, end),
                    "endpoint_idx": end,
                })
                opt[start:end + 1] = [candidate_number] * (end + 1 - start)
            decision_idx = markers[cursor][0]
            cursor += 1
            opt[decision_idx] = OPT_DECIDE
            seg[question_idx:decision_idx + 1] = [question_number] * (decision_idx + 1 - question_idx)

            first_option = candidate_spans[0]["span"][0]
            instruction_length = first_option - question_idx
            pos[question_idx:first_option] = range(first_question, first_question + instruction_length)
            option_start_pos = first_question + instruction_length
            for candidate in candidate_spans:
                start, stop = candidate["span"]
                pos[start:stop] = range(option_start_pos, option_start_pos + stop - start)
            longest = max(stop - start for start, stop in (candidate["span"] for candidate in candidate_spans))
            pos[decision_idx] = option_start_pos + longest

            decisions.append(decision_idx)
            endpoints.append([candidate["endpoint_idx"] for candidate in candidate_spans])
            spans.append({
                "question_span": (question_idx + 1, first_option),
                "question_marker_idx": question_idx,
                "candidates": candidate_spans,
            })
            metadata.append({key: value for key, value in question.items() if key not in {"question", "candidates"}} | {
                "candidate_ids": [candidate["id"] for candidate in question["candidates"]],
            })

        return {
            "ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decisions, "opt_idx": endpoints,
            "state_span": (state_idx + 1, first_question),
            "state_marker_idx": state_idx, "spans": spans,
            "questions": metadata, "marker_ids": dict(self.marker_ids),
        }
