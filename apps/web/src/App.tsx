import { useRef, useState, type FormEvent } from 'react'
import './App.css'
import { askQuestion, QUESTION_MAX_LENGTH } from './api/client'
import { AnswerView, type AskResult } from './components/AnswerView'
import { CheckWorkForm } from './components/CheckWorkForm'
import { ModeSelector, type Mode } from './components/ModeSelector'
import { PracticeForm } from './components/PracticeForm'
import { StatusIndicator } from './components/StatusIndicator'

const MODES: { id: Mode; label: string }[] = [
  { id: 'tutor', label: 'Tutor' },
  { id: 'check', label: 'Check my work' },
  { id: 'practice', label: 'Practice' },
]

// All three modes are implemented. Tutor is a grounded LLM
// explanation; Check my work and Practice are both backed by the
// deterministic SymPy engine, never an LLM judgment of correctness.

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

        {mode === 'tutor' && <TutorForm key="tutor" />}
        {mode === 'check' && <CheckWorkForm key="check" />}
        {mode === 'practice' && <PracticeForm key="practice" />}
      </main>
    </div>
  )
}

export default App
