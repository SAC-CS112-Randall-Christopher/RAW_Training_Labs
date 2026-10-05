"""Strict operator inputs; no executable commands or client-selected artifact paths."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Criteria(Input):
    task: Literal["work_request", "decision", "exact_text"] = "work_request"
    fields: list[str] = Field(default_factory=lambda: ["site", "trade", "urgency", "summary"])
    minimum_success: float = Field(default=0.8, ge=0, le=1)
    maximum_regressions: int = Field(default=0, ge=0, le=64)
    maximum_invalid: int = Field(default=0, ge=0, le=64)
    maximum_p95_seconds: float = Field(default=60.0, gt=0, le=600)
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def field_names(self):
        if not self.fields or len(self.fields) > 32 or len(set(self.fields)) != len(self.fields):
            raise ValueError("Declare 1-32 unique scoring fields")
        if any(not f.isidentifier() or len(f) > 80 for f in self.fields):
            raise ValueError("Scoring fields must be simple names")
        return self


class Project(Input):
    name: str = Field(min_length=1, max_length=160)
    customer: str = Field(min_length=1, max_length=160)
    task: str = Field(min_length=10, max_length=4000)
    criteria: Criteria = Field(default_factory=Criteria)
    approach: Literal["baseline", "retrieval", "tools", "fine_tuning", "combination"] = "baseline"


class Rights(Input):
    training: bool
    evaluation: bool
    external_processing: bool
    sharing: bool
    export: bool
    retention: bool
    basis: str = Field(min_length=10, max_length=2000)


class Message(Input):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=16000)


class Example(Input):
    id: str = Field(min_length=1, max_length=160)
    prompt: list[Message] = Field(min_length=1, max_length=32)
    completion: str = Field(min_length=1, max_length=16000)
    group: str = Field(min_length=1, max_length=160)
    split: Literal["train", "validation", "test", "regression"]
    available_at: int = Field(ge=0, le=4102444800)
    source_kind: Literal["raw_synthetic", "customer", "permitted_general"]
    rights: Rights

    @model_validator(mode="after")
    def valid(self):
        if self.prompt[-1].role == "assistant":
            raise ValueError("Final target must follow an input or tool result")
        if not self.rights.retention:
            raise ValueError("Retention permission is required to import")
        return self


class Import(Input):
    examples: list[Example] = Field(min_length=1, max_length=64)


class Review(Input):
    reviewer: str = Field(min_length=1, max_length=160)
    reviewer_kind: Literal["human", "delegated_semantic"]
    reviewer_authored_material: bool
    approved: bool
    completion: str = Field(min_length=1, max_length=16000)
    reason: str = Field(min_length=10, max_length=2000)
    rights: Rights


class Start(Input):
    kind: Literal["acquire", "baseline", "train", "compare", "export"]
    corpus_id: str | None = None
    profile_id: str = "qwen3-0.6b-fast-cpu"
    authorized: bool = False
    idempotency_key: str = Field(min_length=8, max_length=160)
    baseline_id: str | None = None
    candidate_id: str | None = None
    resume_id: str | None = None
    yield_after_step: int | None = Field(default=None, ge=1, le=5)


class DecisionReview(Input):
    reviewer: str = Field(min_length=1, max_length=160)
    case_id: str = Field(min_length=1, max_length=160)
    assessment: Literal["candidate_better", "baseline_better", "equivalent", "uncertain"]
    reason: str = Field(min_length=10, max_length=2000)
