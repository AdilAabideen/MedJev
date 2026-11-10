"""Typed input requests. Gold answers are kept outside these models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


NonBlank = Annotated[str, AfterValidator(nonblank)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(InputModel):
    id: NonBlank
    text: NonBlank


class Noul(InputModel):
    type: Literal["noul"]
    question: NonBlank


class Choice(InputModel):
    type: Literal["choice"]
    question: NonBlank
    candidates: list[Candidate] = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def unique_candidates(self) -> Choice:
        if len({candidate.id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        return self


class Score(Choice):
    type: Literal["score"]
    ordered_values: list[int | float] = Field(min_length=1, max_length=255)
    scale_id: str | None = None
    score_direction: str | None = None

    @model_validator(mode="after")
    def matching_values(self) -> Score:
        if len(self.ordered_values) != len(self.candidates):
            raise ValueError("ordered_values must match the number of candidates")
        if len(set(self.ordered_values)) != len(self.ordered_values):
            raise ValueError("ordered_values must be unique")
        return self


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class DecisionRequest(InputModel):
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1)

    @model_validator(mode="after")
    def named_questions(self) -> DecisionRequest:
        if any(not name.strip() for name in self.questions):
            raise ValueError("question IDs must not be empty")
        return self
