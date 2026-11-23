import { useState } from 'react'
import { submitFeedback } from '../api/client'
import type { FeedbackRating } from '../api/types'

type SubmitState = 'idle' | 'submitting' | 'submitted' | 'error'

interface FeedbackControlsProps {
  answerId: string
}

/** Mount with `key={answerId}` from the caller so a new answer always
 * gets a fresh, reset instance of this component — simpler and more
 * reliable than manually clearing local state on every question
 * change. */
export function FeedbackControls({ answerId }: FeedbackControlsProps) {
  const [rating, setRating] = useState<FeedbackRating | null>(null)
  const [correction, setCorrection] = useState('')
  const [state, setState] = useState<SubmitState>('idle')

  async function handleSubmit() {
    if (rating === null || state === 'submitting') return
    setState('submitting')
    try {
      await submitFeedback({
        answer_id: answerId,
        rating,
        correction: correction.trim().length > 0 ? correction.trim() : null,
      })
      setState('submitted')
    } catch {
      setState('error')
    }
  }

  if (state === 'submitted') {
    return (
      <div className="feedback feedback-done">
        <p>Thanks for the feedback.</p>
        <button
          type="button"
          className="feedback-link-button"
          onClick={() => setState('idle')}
        >
          Change feedback
        </button>
      </div>
    )
  }

  return (
    <div className="feedback">
      <p className="feedback-prompt">Was this explanation helpful?</p>
      <div className="feedback-buttons" role="group" aria-label="Rate this answer">
        <button
          type="button"
          className={rating === 'up' ? 'feedback-button feedback-button-active' : 'feedback-button'}
          aria-pressed={rating === 'up'}
          onClick={() => setRating('up')}
          disabled={state === 'submitting'}
        >
          Helpful
        </button>
        <button
          type="button"
          className={rating === 'down' ? 'feedback-button feedback-button-active' : 'feedback-button'}
          aria-pressed={rating === 'down'}
          onClick={() => setRating('down')}
          disabled={state === 'submitting'}
        >
          Not helpful
        </button>
      </div>

      {rating !== null && (
        <>
          <textarea
            className="feedback-text"
            placeholder="Optional: what was wrong, or what was good?"
            value={correction}
            onChange={(event) => setCorrection(event.target.value)}
            maxLength={1000}
            rows={3}
            disabled={state === 'submitting'}
          />
          <button
            type="button"
            className="feedback-submit"
            onClick={handleSubmit}
            disabled={state === 'submitting'}
          >
            {state === 'submitting' ? 'Submitting...' : 'Submit feedback'}
          </button>
          {state === 'error' && (
            <p className="feedback-error">Couldn't submit feedback. Please try again.</p>
          )}
        </>
      )}
    </div>
  )
}
