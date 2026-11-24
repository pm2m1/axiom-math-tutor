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

// Mirrors apps/api/axiom/models/verification.py (Phase 2A/2B).

export type VerificationStatus =
  | 'verified_correct'
  | 'verified_incorrect'
  | 'incomplete'
  | 'unsupported'
  | 'indeterminate'

export type Category = 'linear' | 'quadratic' | 'unsupported'

export interface CandidateResult {
  value: string
  valid: boolean | null
}

export interface VerifyResponse {
  status: VerificationStatus
  category: Category
  domain: 'real'
  deterministic: boolean
  reason: string | null
  detail: string
  complete: boolean | null
  candidates: CandidateResult[]
}

export type TransitionKind =
  | 'equivalent'
  | 'narrowing'
  | 'broadening'
  | 'partial_overlap'
  | 'incompatible'
  | 'unsupported'
  | 'indeterminate'

export type StepCheckStatus =
  | 'all_steps_equivalent'
  | 'valid_but_incomplete'
  | 'invalid_step_found'
  | 'unsupported'
  | 'verification_incomplete'

export type MisconceptionLabel =
  | 'lost_solution_branch'
  | 'introduced_extraneous_root'
  | 'sign_error'
  | 'general_error'

export interface StepResult {
  index: number
  equation: string
  category: Category
  transition: TransitionKind
  detail: string
  misconception: MisconceptionLabel | null
}

export interface CheckWorkRequest {
  equation: string
  variable?: string
  steps: string[]
  solutions?: string[] | null
  hint_level?: number | null
}

export interface CheckWorkResponse {
  status: StepCheckStatus
  steps: StepResult[]
  first_invalid_step_index: number | null
  first_non_equivalent_step_index: number | null
  chain_preserves_solution_set: boolean | null
  deterministic: boolean
  detail: string
  final_answer: VerifyResponse | null
  hints: string[] | null
}

// Mirrors apps/api/axiom/models/practice.py (Phase 3).

export type Skill = 'linear_equations' | 'quadratic_equations'
export type Difficulty = 'easy' | 'medium' | 'hard'

export interface NextProblemRequest {
  learner_id: string
}

export interface NextProblemResponse {
  skill: Skill
  difficulty: Difficulty
  equation: string
  variable: string
}

export interface SubmitAnswerRequest {
  learner_id: string
  skill: Skill
  equation: string
  variable?: string
  solutions: string[]
  hints_used?: number
}

export interface SubmitAnswerResponse {
  status: VerificationStatus
  detail: string
  correct: boolean | null
  mastery_estimate: number
}

export interface MasterySummaryRequest {
  learner_id: string
}

export interface SkillMastery {
  skill: string
  estimate: number
}

export interface MasterySummaryResponse {
  skills: SkillMastery[]
}
