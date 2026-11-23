import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { submitFeedback } from '../api/client'
import { FeedbackControls } from './FeedbackControls'

vi.mock('../api/client', () => ({
  submitFeedback: vi.fn(),
}))

afterEach(() => {
  vi.mocked(submitFeedback).mockReset()
})

describe('FeedbackControls', () => {
  it('submits the selected rating and optional correction for the given answer', async () => {
    vi.mocked(submitFeedback).mockResolvedValue({
      feedback_id: 'f-1',
      answer_id: 'answer-1',
      rating: 'up',
      correction: 'nice',
      answer_status: 'kb',
      created_at: '2026-01-01T00:00:00Z',
    })

    render(<FeedbackControls answerId="answer-1" />)

    fireEvent.click(screen.getByRole('button', { name: 'Helpful' }))
    fireEvent.change(screen.getByPlaceholderText(/optional/i), { target: { value: 'nice' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit feedback' }))

    expect(await screen.findByText('Thanks for the feedback.')).toBeInTheDocument()
    expect(submitFeedback).toHaveBeenCalledWith({
      answer_id: 'answer-1',
      rating: 'up',
      correction: 'nice',
    })
  })

  it('sends null, not an empty string, when no correction is entered', async () => {
    vi.mocked(submitFeedback).mockResolvedValue({
      feedback_id: 'f-2',
      answer_id: 'answer-1',
      rating: 'down',
      correction: null,
      answer_status: 'kb',
      created_at: '2026-01-01T00:00:00Z',
    })

    render(<FeedbackControls answerId="answer-1" />)
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit feedback' }))

    await screen.findByText('Thanks for the feedback.')
    expect(submitFeedback).toHaveBeenCalledWith({
      answer_id: 'answer-1',
      rating: 'down',
      correction: null,
    })
  })

  it('shows an error and allows retry when submission fails', async () => {
    vi.mocked(submitFeedback).mockRejectedValue(new Error('network down'))

    render(<FeedbackControls answerId="answer-1" />)
    fireEvent.click(screen.getByRole('button', { name: 'Helpful' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit feedback' }))

    expect(await screen.findByText(/couldn't submit feedback/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit feedback' })).not.toBeDisabled()
  })

  it('does not render a submit control until a rating is chosen', () => {
    render(<FeedbackControls answerId="answer-1" />)
    expect(screen.queryByRole('button', { name: 'Submit feedback' })).not.toBeInTheDocument()
  })
})
