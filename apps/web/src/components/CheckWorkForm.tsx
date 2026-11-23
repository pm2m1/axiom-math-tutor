import { useRef, useState, type FormEvent } from 'react'
import { checkWork, EQUATION_MAX_LENGTH, MAX_SOLUTIONS, MAX_STEPS } from '../api/client'
import type {
  CheckWorkResponse,
  TransitionKind,
  VerificationStatus,
  VerifyResponse,
} from '../api/types'

type RequestState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'success'; response: CheckWorkResponse }
  | { kind: 'error'; message: string }

interface Row {
  id: number
  value: string
}

let nextRowId = 0
function newRow(): Row {
  nextRowId += 1
  return { id: nextRowId, value: '' }
}

const TRANSITION_LABEL: Record<TransitionKind, string> = {
  equivalent: 'Equivalent',
  narrowing: 'Narrows the solution set',
  broadening: 'Broadens the solution set',
  partial_overlap: 'Partially overlapping solution set',
  incompatible: 'Incompatible — provably incorrect',
  unsupported: 'Unsupported',
  indeterminate: 'Indeterminate',
}

const STATUS_LABEL: Record<CheckWorkResponse['status'], string> = {
  all_steps_equivalent: 'All steps preserve the solution set',
  valid_but_incomplete: 'No error found, but not every step is fully equivalent',
  invalid_step_found: 'A provably incorrect step was found',
  unsupported: 'This equation is outside the supported grammar',
  verification_incomplete: 'Verification was incomplete for part of this chain',
}

const FINAL_STATUS_LABEL: Record<VerificationStatus, string> = {
  verified_correct: 'Correct and complete',
  verified_incorrect: 'Incorrect',
  incomplete: 'Valid but incomplete',
  unsupported: 'Unsupported',
  indeterminate: 'Indeterminate',
}

/** Student-facing Check My Work (Phase 2B) — posts to `/api/v1/check-work`,
 * a deterministic SymPy check, never an LLM judgment. Rating Tutor-mode
 * explanations as "verified" is explicitly not a claim this makes. */
export function CheckWorkForm() {
  const [equation, setEquation] = useState('')
  const [steps, setSteps] = useState<Row[]>([])
  const [solutionsMode, setSolutionsMode] = useState<'omitted' | 'listed'>('omitted')
  const [solutions, setSolutions] = useState<Row[]>([])
  const [state, setState] = useState<RequestState>({ kind: 'idle' })
  const abortRef = useRef<AbortController | null>(null)

  const loading = state.kind === 'loading'
  const submitDisabled = loading || equation.trim().length === 0

  function addStep() {
    if (steps.length >= MAX_STEPS) return
    setSteps((prev) => [...prev, newRow()])
  }
  function removeStep(id: number) {
    setSteps((prev) => prev.filter((s) => s.id !== id))
  }
  function updateStep(id: number, value: string) {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, value } : s)))
  }

  function addSolution() {
    if (solutions.length >= MAX_SOLUTIONS) return
    setSolutionsMode('listed')
    setSolutions((prev) => [...prev, newRow()])
  }
  function removeSolution(id: number) {
    setSolutions((prev) => prev.filter((s) => s.id !== id))
  }
  function updateSolution(id: number, value: string) {
    setSolutions((prev) => prev.map((s) => (s.id === id ? { ...s, value } : s)))
  }
  function claimNoRealSolutions() {
    setSolutionsMode('listed')
    setSolutions([])
  }
  function clearFinalAnswer() {
    setSolutionsMode('omitted')
    setSolutions([])
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (submitDisabled) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setState({ kind: 'loading' })
    try {
      const response = await checkWork(
        {
          equation,
          steps: steps.map((s) => s.value),
          solutions: solutionsMode === 'omitted' ? undefined : solutions.map((s) => s.value),
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setState({ kind: 'success', response })
    } catch {
      if (controller.signal.aborted) return
      setState({ kind: 'error', message: 'Could not reach the server. Please try again.' })
    }
  }

  return (
    <div className="check-work">
      <form className="check-work-form" onSubmit={handleSubmit}>
        <label className="check-work-label" htmlFor="check-work-equation">
          Original equation
        </label>
        <input
          id="check-work-equation"
          className="check-work-input"
          type="text"
          placeholder="e.g. 2*x + 3 = 11"
          value={equation}
          onChange={(event) => setEquation(event.target.value)}
          maxLength={EQUATION_MAX_LENGTH}
          disabled={loading}
        />

        <div className="check-work-steps">
          <p className="check-work-label">Your steps (optional)</p>
          {steps.map((step, i) => (
            <div className="check-work-row" key={step.id}>
              <span className="check-work-step-number" aria-hidden="true">
                {i + 1}
              </span>
              <input
                className="check-work-input"
                type="text"
                aria-label={`Step ${i + 1}`}
                value={step.value}
                onChange={(event) => updateStep(step.id, event.target.value)}
                maxLength={EQUATION_MAX_LENGTH}
                disabled={loading}
              />
              <button
                type="button"
                className="check-work-remove-button"
                aria-label={`Remove step ${i + 1}`}
                onClick={() => removeStep(step.id)}
                disabled={loading}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            className="check-work-add-button"
            onClick={addStep}
            disabled={loading || steps.length >= MAX_STEPS}
          >
            Add step
          </button>
        </div>

        <div className="check-work-solutions">
          <p className="check-work-label">Final answer (optional)</p>
          {solutionsMode === 'listed' ? (
            <>
              {solutions.map((solution, i) => (
                <div className="check-work-row" key={solution.id}>
                  <input
                    className="check-work-input"
                    type="text"
                    aria-label={`Proposed solution ${i + 1}`}
                    value={solution.value}
                    onChange={(event) => updateSolution(solution.id, event.target.value)}
                    maxLength={EQUATION_MAX_LENGTH}
                    disabled={loading}
                  />
                  <button
                    type="button"
                    className="check-work-remove-button"
                    aria-label={`Remove solution ${i + 1}`}
                    onClick={() => removeSolution(solution.id)}
                    disabled={loading}
                  >
                    Remove
                  </button>
                </div>
              ))}
              {solutions.length === 0 && (
                <p className="check-work-note">Claiming this equation has no real solutions.</p>
              )}
              <div className="check-work-solutions-actions">
                <button
                  type="button"
                  className="check-work-add-button"
                  onClick={addSolution}
                  disabled={loading || solutions.length >= MAX_SOLUTIONS}
                >
                  Add solution
                </button>
                <button
                  type="button"
                  className="check-work-link-button"
                  onClick={clearFinalAnswer}
                  disabled={loading}
                >
                  Don't check a final answer
                </button>
              </div>
            </>
          ) : (
            <div className="check-work-solutions-actions">
              <button
                type="button"
                className="check-work-add-button"
                onClick={addSolution}
                disabled={loading}
              >
                Add solution
              </button>
              <button
                type="button"
                className="check-work-link-button"
                onClick={claimNoRealSolutions}
                disabled={loading}
              >
                This equation has no real solutions
              </button>
            </div>
          )}
        </div>

        <button type="submit" className="ask-button" disabled={submitDisabled}>
          {loading ? 'Checking…' : 'Check my work'}
        </button>
      </form>

      {state.kind === 'error' && <p className="check-work-error">{state.message}</p>}
      {state.kind === 'success' && <CheckWorkResultView response={state.response} />}
    </div>
  )
}

function CheckWorkResultView({ response }: { response: CheckWorkResponse }) {
  return (
    <div className="check-work-result">
      <p className="answer-status-label">{STATUS_LABEL[response.status]}</p>
      <p>{response.detail}</p>

      {response.steps.length > 0 && (
        <ol className="check-work-step-list">
          {response.steps.map((step) => {
            const isFirstInvalid = step.index === response.first_invalid_step_index
            return (
              <li
                key={step.index}
                className={
                  isFirstInvalid
                    ? 'check-work-step-result check-work-step-result-invalid'
                    : `check-work-step-result check-work-step-result-${step.transition}`
                }
              >
                <p className="check-work-step-equation">
                  Step {step.index}: {step.equation}
                </p>
                <p className="check-work-step-transition">
                  {isFirstInvalid && (
                    <span className="check-work-first-error-flag">First error — </span>
                  )}
                  {TRANSITION_LABEL[step.transition]}
                </p>
                <p className="check-work-step-detail">{step.detail}</p>
              </li>
            )
          })}
        </ol>
      )}

      {response.final_answer && <FinalAnswerResultView result={response.final_answer} />}
    </div>
  )
}

function FinalAnswerResultView({ result }: { result: VerifyResponse }) {
  return (
    <div className="check-work-final-answer">
      <p className="answer-status-label">Final answer: {FINAL_STATUS_LABEL[result.status]}</p>
      <p>{result.detail}</p>
      {result.candidates.length > 0 && (
        <ul className="check-work-candidate-list">
          {result.candidates.map((candidate) => (
            <li key={candidate.value}>
              {candidate.value}:{' '}
              {candidate.valid === null
                ? 'indeterminate'
                : candidate.valid
                  ? 'valid'
                  : 'invalid'}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
