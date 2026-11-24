import { useEffect, useState } from 'react'
import {
  checkWork,
  EQUATION_MAX_LENGTH,
  masterySummary,
  MAX_SOLUTIONS,
  nextPracticeProblem,
  submitPracticeAnswer,
} from '../api/client'
import type { NextProblemResponse, SkillMastery, SubmitAnswerResponse } from '../api/types'

const LEARNER_ID_STORAGE_KEY = 'axiom-learner-id'
const MAX_HINT_LEVEL = 4

const SKILL_LABEL: Record<string, string> = {
  linear_equations: 'Linear equations',
  quadratic_equations: 'Quadratic equations',
}

const DIFFICULTY_LABEL: Record<NextProblemResponse['difficulty'], string> = {
  easy: 'Easy',
  medium: 'Medium',
  hard: 'Hard',
}

/** A pseudonymous, client-generated id — not an account, not
 * authenticated. Stored in `localStorage` only so mastery persists
 * across visits in *this* browser; it proves nothing about identity
 * across devices or sessions. See docs/jee-solving.md. */
function getOrCreateLearnerId(): string {
  try {
    const existing = window.localStorage.getItem(LEARNER_ID_STORAGE_KEY)
    if (existing) return existing
    const created =
      typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `learner-${Math.random().toString(36).slice(2)}`
    window.localStorage.setItem(LEARNER_ID_STORAGE_KEY, created)
    return created
  } catch {
    // localStorage unavailable (e.g. disabled) — fall back to a
    // per-mount id; mastery just won't persist across reloads.
    return `learner-${Math.random().toString(36).slice(2)}`
  }
}

interface Row {
  id: number
  value: string
}

let nextRowId = 0
function newRow(): Row {
  nextRowId += 1
  return { id: nextRowId, value: '' }
}

type ProblemState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'ready'; problem: NextProblemResponse }
  | { kind: 'error' }

type SubmitState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; result: SubmitAnswerResponse }
  | { kind: 'error' }

/** Student-facing Practice (Phase 3) — every generated problem is
 * independently re-verified server-side before being shown, and every
 * submitted answer is graded by the same deterministic engine
 * `/api/v1/verify` uses, never an LLM. */
export function PracticeForm() {
  const [learnerId] = useState<string>(getOrCreateLearnerId)

  const [mastery, setMastery] = useState<SkillMastery[] | null>(null)
  const [problemState, setProblemState] = useState<ProblemState>({ kind: 'idle' })
  const [answers, setAnswers] = useState<Row[]>([newRow()])
  const [submitState, setSubmitState] = useState<SubmitState>({ kind: 'idle' })
  const [hints, setHints] = useState<string[] | null>(null)
  const [hintLoading, setHintLoading] = useState(false)
  const [hintError, setHintError] = useState(false)

  useEffect(() => {
    masterySummary({ learner_id: learnerId })
      .then((response) => setMastery(response.skills))
      .catch(() => setMastery(null))
  }, [learnerId])

  const problem = problemState.kind === 'ready' ? problemState.problem : null
  const hintLevel = hints?.length ?? 0
  const submitDisabled =
    submitState.kind === 'loading' ||
    problem === null ||
    answers.every((a) => a.value.trim().length === 0)

  async function loadNextProblem() {
    setProblemState({ kind: 'loading' })
    setSubmitState({ kind: 'idle' })
    setAnswers([newRow()])
    setHints(null)
    setHintError(false)
    try {
      const response = await nextPracticeProblem({ learner_id: learnerId })
      setProblemState({ kind: 'ready', problem: response })
    } catch {
      setProblemState({ kind: 'error' })
    }
  }

  function addAnswer() {
    if (answers.length >= MAX_SOLUTIONS) return
    setAnswers((prev) => [...prev, newRow()])
  }
  function removeAnswer(id: number) {
    setAnswers((prev) => (prev.length > 1 ? prev.filter((a) => a.id !== id) : prev))
  }
  function updateAnswer(id: number, value: string) {
    setAnswers((prev) => prev.map((a) => (a.id === id ? { ...a, value } : a)))
  }

  async function handleSubmit() {
    if (submitDisabled || !problem) return
    setSubmitState({ kind: 'loading' })
    try {
      const response = await submitPracticeAnswer({
        learner_id: learnerId,
        skill: problem.skill,
        equation: problem.equation,
        variable: problem.variable,
        solutions: answers.map((a) => a.value).filter((v) => v.trim().length > 0),
        hints_used: hintLevel,
      })
      setSubmitState({ kind: 'done', result: response })
      setMastery((prev) =>
        prev
          ? prev.map((s) =>
              s.skill === problem.skill ? { ...s, estimate: response.mastery_estimate } : s,
            )
          : prev,
      )
    } catch {
      setSubmitState({ kind: 'error' })
    }
  }

  async function requestNextHint() {
    if (!problem || hintLoading || hintLevel >= MAX_HINT_LEVEL) return
    setHintLoading(true)
    setHintError(false)
    try {
      const response = await checkWork({
        equation: problem.equation,
        variable: problem.variable,
        steps: [],
        hint_level: hintLevel + 1,
      })
      setHints(response.hints)
    } catch {
      setHintError(true)
    } finally {
      setHintLoading(false)
    }
  }

  return (
    <div className="practice">
      {mastery && (
        <div className="practice-mastery">
          <p className="check-work-label">Your mastery</p>
          <ul className="practice-mastery-list">
            {mastery.map((skill) => (
              <li key={skill.skill}>
                {SKILL_LABEL[skill.skill] ?? skill.skill}:{' '}
                {Math.round(skill.estimate * 100)}%
              </li>
            ))}
          </ul>
        </div>
      )}

      {problemState.kind === 'idle' && (
        <button type="button" className="ask-button" onClick={loadNextProblem}>
          Start practicing
        </button>
      )}

      {problemState.kind === 'loading' && <p>Loading a problem…</p>}

      {problemState.kind === 'error' && (
        <div>
          <p className="check-work-error">Could not load a practice problem. Please try again.</p>
          <button type="button" className="ask-button" onClick={loadNextProblem}>
            Try again
          </button>
        </div>
      )}

      {problem && (
        <div className="practice-problem">
          <p className="check-work-label">
            {SKILL_LABEL[problem.skill] ?? problem.skill} — {DIFFICULTY_LABEL[problem.difficulty]}
          </p>
          <p className="practice-equation">{problem.equation}</p>

          <div className="check-work-steps">
            <p className="check-work-label">Your answer</p>
            {answers.map((answer, i) => (
              <div className="check-work-row" key={answer.id}>
                <input
                  className="check-work-input"
                  type="text"
                  aria-label={`Answer ${i + 1}`}
                  value={answer.value}
                  onChange={(event) => updateAnswer(answer.id, event.target.value)}
                  maxLength={EQUATION_MAX_LENGTH}
                  disabled={submitState.kind === 'loading'}
                />
                {answers.length > 1 && (
                  <button
                    type="button"
                    className="check-work-remove-button"
                    aria-label={`Remove answer ${i + 1}`}
                    onClick={() => removeAnswer(answer.id)}
                    disabled={submitState.kind === 'loading'}
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            <button
              type="button"
              className="check-work-add-button"
              onClick={addAnswer}
              disabled={submitState.kind === 'loading' || answers.length >= MAX_SOLUTIONS}
            >
              Add another value
            </button>
          </div>

          <div className="check-work-hints">
            <button
              type="button"
              className="check-work-add-button"
              onClick={requestNextHint}
              disabled={hintLoading || hintLevel >= MAX_HINT_LEVEL}
            >
              {hintLoading ? 'Loading hint…' : hintLevel === 0 ? 'Get a hint' : 'Get another hint'}
            </button>
            {hintError && <p className="check-work-error">Could not load a hint.</p>}
            {hints && (
              <ol className="check-work-hint-list">
                {hints.map((hint, i) => (
                  <li key={i}>{hint}</li>
                ))}
              </ol>
            )}
          </div>

          <button
            type="button"
            className="ask-button"
            onClick={handleSubmit}
            disabled={submitDisabled}
          >
            {submitState.kind === 'loading' ? 'Checking…' : 'Submit answer'}
          </button>

          {submitState.kind === 'error' && (
            <p className="check-work-error">Could not reach the server. Please try again.</p>
          )}

          {submitState.kind === 'done' && (
            <div className="practice-result">
              <p
                className={
                  submitState.result.correct
                    ? 'practice-result-correct'
                    : submitState.result.correct === false
                      ? 'practice-result-incorrect'
                      : 'practice-result-unknown'
                }
              >
                {submitState.result.correct === true && 'Correct!'}
                {submitState.result.correct === false && 'Not quite.'}
                {submitState.result.correct === null && 'Could not verify this answer.'}
              </p>
              <p>{submitState.result.detail}</p>
              <button type="button" className="ask-button" onClick={loadNextProblem}>
                Next problem
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
