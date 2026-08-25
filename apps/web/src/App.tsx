import { useState } from 'react'
import './App.css'
import { ModeSelector, type Mode } from './components/ModeSelector'
import { StatusIndicator } from './components/StatusIndicator'

const MODES: { id: Mode; label: string }[] = [
  { id: 'tutor', label: 'Tutor' },
  { id: 'check', label: 'Check my work' },
  { id: 'practice', label: 'Practice' },
]

function App() {
  const [mode, setMode] = useState<Mode>('tutor')
  const [question, setQuestion] = useState('')
  const [submittedMode, setSubmittedMode] = useState<Mode | null>(null)

  return (
    <div className="page">
      <header className="topbar">
        <span className="wordmark">Axiom</span>
        <StatusIndicator />
      </header>

      <main className="content">
        <p className="tagline">Verified mathematical tutoring</p>

        <ModeSelector modes={MODES} active={mode} onSelect={setMode} />

        <form
          className="question-form"
          onSubmit={(event) => {
            event.preventDefault()
            setSubmittedMode(mode)
          }}
        >
          <textarea
            className="question-input"
            placeholder="Enter a problem, or paste your attempted solution..."
            value={question}
            onChange={(event) => {
              setQuestion(event.target.value)
              setSubmittedMode(null)
            }}
            rows={5}
          />
          <button type="submit" className="ask-button" disabled={question.trim().length === 0}>
            Ask a question
          </button>
        </form>

        {submittedMode !== null && (
          <p className="placeholder-note">
            No reasoning backend is wired up yet. This would have been routed to the{' '}
            <strong>{MODES.find((m) => m.id === submittedMode)?.label}</strong> mode.
          </p>
        )}
      </main>
    </div>
  )
}

export default App
