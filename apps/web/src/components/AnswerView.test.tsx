import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { AnswerResponse } from '../api/types'
import { AnswerView, type AskResult } from './AnswerView'

vi.mock('./FeedbackControls', () => ({
  FeedbackControls: ({ answerId }: { answerId: string }) => (
    <div data-testid="feedback-controls">{answerId}</div>
  ),
}))

function kbResponse(overrides: Partial<AnswerResponse> = {}): AnswerResponse {
  return {
    status: 'kb',
    answer: 'The derivative of x^2 is 2x.',
    citations: [{ source_id: 'kb-1', source_type: 'kb', title: 'Power rule', url: 'https://example.com/power-rule' }],
    reason: null,
    detail: null,
    answer_id: 'answer-1',
    ...overrides,
  }
}

describe('AnswerView', () => {
  it('renders a knowledge-base answer with citations and feedback controls', () => {
    const result: AskResult = { kind: 'success', response: kbResponse() }
    render(<AnswerView result={result} />)

    expect(screen.getByText('From the knowledge base')).toBeInTheDocument()
    expect(screen.getByText('The derivative of x^2 is 2x.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Power rule' })).toHaveAttribute(
      'href',
      'https://example.com/power-rule',
    )
    expect(screen.getByTestId('feedback-controls')).toHaveTextContent('answer-1')
  })

  it('renders a web-sourced answer distinctly from a kb answer', () => {
    const result: AskResult = {
      kind: 'success',
      response: kbResponse({
        status: 'web',
        citations: [{ source_id: 'w-1', source_type: 'web', title: 'Web source', url: 'https://example.com' }],
      }),
    }
    render(<AnswerView result={result} />)

    expect(screen.getByText('From a web search')).toBeInTheDocument()
  })

  it('does not render citations or feedback controls when there are none', () => {
    const result: AskResult = {
      kind: 'success',
      response: kbResponse({ citations: [], answer_id: null }),
    }
    render(<AnswerView result={result} />)

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByTestId('feedback-controls')).not.toBeInTheDocument()
  })

  it('renders a refusal with its reason, and no feedback controls', () => {
    const result: AskResult = {
      kind: 'success',
      response: kbResponse({
        status: 'refused',
        answer: null,
        citations: [],
        reason: 'off_topic',
        answer_id: null,
      }),
    }
    render(<AnswerView result={result} />)

    expect(screen.getByText('Not processed')).toBeInTheDocument()
    expect(screen.getByText("This doesn't look like a mathematics question.")).toBeInTheDocument()
    expect(screen.queryByTestId('feedback-controls')).not.toBeInTheDocument()
  })

  it('renders an abstention', () => {
    const result: AskResult = {
      kind: 'success',
      response: kbResponse({
        status: 'abstained',
        answer: null,
        citations: [],
        reason: 'insufficient_evidence',
        answer_id: null,
      }),
    }
    render(<AnswerView result={result} />)

    expect(screen.getByText('Unable to answer')).toBeInTheDocument()
    expect(screen.getByText("There isn't enough information to answer this responsibly.")).toBeInTheDocument()
  })

  it('renders a backend-reported dependency failure', () => {
    const result: AskResult = {
      kind: 'success',
      response: kbResponse({
        status: 'unavailable',
        answer: null,
        citations: [],
        reason: 'retrieval_error',
        answer_id: null,
      }),
    }
    render(<AnswerView result={result} />)

    expect(screen.getByText('Service unavailable')).toBeInTheDocument()
    expect(screen.getByText('The knowledge base is temporarily unavailable.')).toBeInTheDocument()
  })

  it('renders a frontend-side network error distinctly from a backend status', () => {
    const result: AskResult = { kind: 'network-error', message: 'Could not reach the server. Please try again.' }
    render(<AnswerView result={result} />)

    expect(screen.getByText('Connection problem')).toBeInTheDocument()
    expect(screen.getByText('Could not reach the server. Please try again.')).toBeInTheDocument()
  })
})
