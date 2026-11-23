from typing import Literal

from pydantic import BaseModel, Field

Rating = Literal["up", "down"]
MAX_CORRECTION_LENGTH = 1000


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=100)
    rating: Rating
    correction: str | None = Field(default=None, max_length=MAX_CORRECTION_LENGTH)


class FeedbackResponse(BaseModel):
    feedback_id: str
    answer_id: str
    rating: Rating
    correction: str | None
    answer_status: str
    created_at: str
