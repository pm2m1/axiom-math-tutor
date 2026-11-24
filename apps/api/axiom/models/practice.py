from typing import Literal

from pydantic import BaseModel, Field

from axiom.models.verification import VerificationStatus

Skill = Literal["linear_equations", "quadratic_equations"]
Difficulty = Literal["easy", "medium", "hard"]

_LearnerId = Field(min_length=1, max_length=100)


class NextProblemRequest(BaseModel):
    """`learner_id` is an opaque, client-generated pseudonymous id (e.g.
    a random id stored in browser `localStorage`) — not an account,
    not authenticated. See `axiom/mastery.py`."""

    learner_id: str = _LearnerId


class NextProblemResponse(BaseModel):
    skill: Skill
    difficulty: Difficulty
    equation: str
    variable: str


class SubmitAnswerRequest(BaseModel):
    learner_id: str = _LearnerId
    skill: Skill
    equation: str = Field(min_length=1, max_length=200)
    variable: str = Field(default="x", min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    solutions: list[str] = Field(default_factory=list, max_length=20)
    hints_used: int = Field(default=0, ge=0, le=4)


class SubmitAnswerResponse(BaseModel):
    """`correct` is `None` whenever the verification itself was
    `"unsupported"`/`"indeterminate"` — mastery is only ever updated
    from a decisive `True`/`False`, documented in `axiom/mastery.py`."""

    status: VerificationStatus
    detail: str
    correct: bool | None
    mastery_estimate: float


class MasterySummaryRequest(BaseModel):
    learner_id: str = _LearnerId


class SkillMastery(BaseModel):
    skill: str
    estimate: float


class MasterySummaryResponse(BaseModel):
    skills: list[SkillMastery]
