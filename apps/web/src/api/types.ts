// Mirrors apps/api/axiom/models/answer.py and feedback.py. Keep these in
// sync by hand — there is no shared schema generation in this project.

export type AnswerStatus = 'kb' | 'web' | 'refused' | 'abstained' | 'unavailable'

export type AnswerReason =
  | 'off_topic'
  | 'prompt_injection'
  | 'insufficient_evidence'
  | 'invalid_output'
  | 'retrieval_error'
  | 'judge_error'
  | 'search_error'
  | 'generation_error'

export interface Citation {
  source_id: string
  source_type: 'kb' | 'web'
  title: string
  url: string
}

export interface AnswerResponse {
  status: AnswerStatus
  answer: string | null
  citations: Citation[]
  reason: AnswerReason | null
  detail: string | null
  answer_id: string | null
}

export type FeedbackRating = 'up' | 'down'

export interface FeedbackRequest {
  answer_id: string
  rating: FeedbackRating
  correction?: string | null
}

export interface FeedbackResponse {
  feedback_id: string
  answer_id: string
  rating: FeedbackRating
  correction: string | null
  answer_status: string
  created_at: string
}
