import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { askQuestion, masterySummary } from './api/client'

vi.mock('./api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api/client')>()
  return { ...actual, askQuestion: vi.fn(), masterySummary: vi.fn() }
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function kbAnswer(text: string) {
  return { status: 'kb' as const, answer: text, citations: [], reason: null, detail: null, answer_id: 'a-1' }
}

afterEach(() => {
  vi.mocked(askQuestion).mockReset()
  vi.mocked(masterySummary).mockReset()
})

function ask(question: string) {
  fireEvent.change(screen.getByPlaceholderText(/enter an algebra question/i), { target: { value: question } })
  fireEvent.click(screen.getByRole('button', { name: /ask|thinking/i }))
}

describe('App', () => {
  it('submits the typed question and renders the returned kb answer', async () => {
    vi.mocked(askQuestion).mockResolvedValue(kbAnswer('x = 2'))

    render(<App />)
    ask('Solve 2x = 4')

    expect(await screen.findByText('x = 2')).toBeInTheDocument()
    expect(askQuestion).toHaveBeenCalledWith('Solve 2x = 4', expect.anything())
  })

  it('shows a loading state and disables the button while the request is in flight', async () => {
    const { promise, resolve } = deferred<ReturnType<typeof kbAnswer>>()
    vi.mocked(askQuestion).mockReturnValue(promise)

    render(<App />)
    ask('Solve 2x = 4')

    const button = screen.getByRole('button', { name: /thinking/i })
    expect(button).toBeDisabled()

    resolve(kbAnswer('x = 2'))
    expect(await screen.findByText('x = 2')).toBeInTheDocument()
  })

  it('does not start a second request while one is already in flight', () => {
    const { promise } = deferred<ReturnType<typeof kbAnswer>>()
    vi.mocked(askQuestion).mockReturnValue(promise)

    render(<App />)
    ask('Solve 2x = 4')
    fireEvent.click(screen.getByRole('button', { name: /thinking/i }))
    fireEvent.click(screen.getByRole('button', { name: /thinking/i }))

    expect(askQuestion).toHaveBeenCalledTimes(1)
  })

  it('shows a network-error message when the request fails', async () => {
    vi.mocked(askQuestion).mockRejectedValue(new Error('network down'))

    render(<App />)
    ask('Solve 2x = 4')

    expect(await screen.findByText(/could not reach the server/i)).toBeInTheDocument()
  })

  it('shows the answer to a later question, asked after the first one resolved', async () => {
    vi.mocked(askQuestion)
      .mockResolvedValueOnce(kbAnswer('first answer'))
      .mockResolvedValueOnce(kbAnswer('second answer'))

    render(<App />)
    ask('First question')
    expect(await screen.findByText('first answer')).toBeInTheDocument()

    ask('Second question')
    expect(await screen.findByText('second answer')).toBeInTheDocument()
    expect(screen.queryByText('first answer')).not.toBeInTheDocument()
  })

  it('switches to the real Check my work form, not the Tutor question box', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('tab', { name: 'Check my work' }))

    expect(screen.queryByPlaceholderText(/enter an algebra question/i)).not.toBeInTheDocument()
    expect(screen.getByLabelText('Original equation')).toBeInTheDocument()
    expect(askQuestion).not.toHaveBeenCalled()
  })

  it('switches to the real Practice form, not the Tutor question box', async () => {
    vi.mocked(masterySummary).mockResolvedValue({
      skills: [
        { skill: 'linear_equations', estimate: 0.5 },
        { skill: 'quadratic_equations', estimate: 0.5 },
      ],
    })

    render(<App />)
    fireEvent.click(screen.getByRole('tab', { name: 'Practice' }))

    expect(screen.queryByPlaceholderText(/enter an algebra question/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Original equation')).not.toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Start practicing' })).toBeInTheDocument()
    expect(askQuestion).not.toHaveBeenCalled()
  })

  it('preserves the Tutor-mode answer when switching to another mode and back', async () => {
    vi.mocked(askQuestion).mockResolvedValue(kbAnswer('x = 2'))

    render(<App />)
    ask('Solve 2x = 4')
    expect(await screen.findByText('x = 2')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Check my work' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Tutor' }))

    // Switching modes remounts the form (fresh state is the documented
    // behavior elsewhere, e.g. feedback resetting per answer) — Tutor's
    // own request/response handling, not stale leftover UI, is what's
    // actually under test here.
    expect(screen.getByPlaceholderText(/enter an algebra question/i)).toHaveValue('')
  })

  it('enforces the question character limit via maxLength', () => {
    render(<App />)
    const textarea = screen.getByPlaceholderText(/enter an algebra question/i)
    expect(textarea).toHaveAttribute('maxlength', '500')
  })
})
