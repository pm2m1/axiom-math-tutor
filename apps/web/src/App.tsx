import { useRef, useState, type FormEvent } from 'react'
import './App.css'
import { askQuestion, QUESTION_MAX_LENGTH } from './api/client'
import { AnswerView, type AskResult } from './components/AnswerView'
import { CheckWorkForm } from './components/CheckWorkForm'
import { ModeSelector, type Mode } from './components/ModeSelector'
import { StatusIndicator } from './components/StatusIndicator'

const MODES: { id: Mode; label: string }[] = [
  { id: 'tutor', label: 'Tutor' },
  { id: 'check', label: 'Check my work' },
  { id: 'practice', label: 'Practice' },
]

// Tutor (grounded explanation) and Check my work (deterministic SymPy
// step verification, Phase 2B) are implemented. Practice is a real
// future mode, not silently routed to an existing backend — shown
// disabled with an honest reason rather than faked.
const UPCOMING_NOTE: Record<Mode, string | null> = {
  tutor: null,
  check: null,
  practice: 'Practice needs adaptive problem generation (Phase 3) and is not available yet.',
}

function TutorForm() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AskResult | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const submitDisabled = loading || question.trim().length === 0

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (submitDisabled) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setResult(null)
    try {
      const response = await askQuestion(question, controller.signal)
      if (controller.signal.aborted) return
      setResult({ kind: 'success', response })
    } catch {
      if (controller.signal.aborted) return
      setResult({ kind: 'network-error', message: 'Could not reach the server. Please try again.' })
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }

  return (
    <>
      <form className="question-form" onSubmit={handleSubmit}>
        <textarea
          className="question-input"
          placeholder="Enter an algebra question..."
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={5}
          maxLength={QUESTION_MAX_LENGTH}
        />
        <div className="question-form-footer">
          <span className="char-count">
            {question.length}/{QUESTION_MAX_LENGTH}
          </span>
          <button type="submit" className="ask-button" disabled={submitDisabled}>
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </div>
      </form>

      {result && <AnswerView result={result} />}
    </>
  )
}

function App() {
  const [mode, setMode] = useState<Mode>('tutor')

  const upcomingNote = UPCOMING_NOTE[mode]

  function selectMode(nextMode: Mode) {
    setMode(nextMode)
  }

  return (
    <div className="page">
      <header className="topbar">
        <span className="wordmark">Axiom</span>
        <StatusIndicator />
      </header>

      <main className="content">
        <p className="tagline">Grounded mathematical explanations</p>

        <ModeSelector modes={MODES} active={mode} onSelect={selectMode} />

        {upcomingNote && <p className="mode-upcoming-note">{upcomingNote}</p>}

        {mode === 'tutor' && <TutorForm key="tutor" />}
        {mode === 'check' && <CheckWorkForm key="check" />}
      </main>
    </div>
  )
}

export default App
