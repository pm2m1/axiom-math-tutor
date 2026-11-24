import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { checkWork, masterySummary, nextPracticeProblem, submitPracticeAnswer } from '../api/client'
import { PracticeForm } from './PracticeForm'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    masterySummary: vi.fn(),
    nextPracticeProblem: vi.fn(),
    submitPracticeAnswer: vi.fn(),
    checkWork: vi.fn(),
  }
})

const defaultMastery = {
  skills: [
    { skill: 'linear_equations', estimate: 0.5 },
    { skill: 'quadratic_equations', estimate: 0.5 },
  ],
}

const sampleProblem = {
  skill: 'linear_equations' as const,
  difficulty: 'easy' as const,
  equation: '2*x + 1 = 7',
  variable: 'x',
}

beforeEach(() => {
  vi.mocked(masterySummary).mockResolvedValue(defaultMastery)
})

afterEach(() => {
  vi.mocked(masterySummary).mockReset()
  vi.mocked(nextPracticeProblem).mockReset()
  vi.mocked(submitPracticeAnswer).mockReset()
  vi.mocked(checkWork).mockReset()
  window.localStorage.clear()
})

async function startPractice() {
  render(<PracticeForm />)
  await screen.findByText(/your mastery/i)
  fireEvent.click(screen.getByRole('button', { name: 'Start practicing' }))
  await screen.findByText(sampleProblem.equation)
}

describe('PracticeForm', () => {
  it('shows the mastery summary on mount', async () => {
    render(<PracticeForm />)

    expect(await screen.findByText(/linear equations/i)).toBeInTheDocument()
    expect(screen.getAllByText(/50%/)).toHaveLength(2)
  })

  it('loads and displays a practice problem on request', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)

    await startPractice()

    expect(screen.getByText(sampleProblem.equation)).toBeInTheDocument()
    expect(nextPracticeProblem).toHaveBeenCalledWith({ learner_id: expect.any(String) })
  })

  it('shows an error and allows retry when problem loading fails', async () => {
    vi.mocked(nextPracticeProblem).mockRejectedValue(new Error('network down'))

    render(<PracticeForm />)
    await screen.findByText(/your mastery/i)
    fireEvent.click(screen.getByRole('button', { name: 'Start practicing' }))

    expect(await screen.findByText(/could not load a practice problem/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('submits the entered answer and shows a correct result', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)
    vi.mocked(submitPracticeAnswer).mockResolvedValue({
      status: 'verified_correct',
      detail: 'the proposed solution set matches the exact real solution set',
      correct: true,
      mastery_estimate: 0.65,
    })

    await startPractice()
    fireEvent.change(screen.getByLabelText('Answer 1'), { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit answer' }))

    expect(await screen.findByText('Correct!')).toBeInTheDocument()
    expect(submitPracticeAnswer).toHaveBeenCalledWith({
      learner_id: expect.any(String),
      skill: 'linear_equations',
      equation: sampleProblem.equation,
      variable: 'x',
      solutions: ['3'],
      hints_used: 0,
    })
  })

  it('shows an incorrect result distinctly from a correct one', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)
    vi.mocked(submitPracticeAnswer).mockResolvedValue({
      status: 'verified_incorrect',
      detail: 'at least one proposed value does not satisfy the equation',
      correct: false,
      mastery_estimate: 0.35,
    })

    await startPractice()
    fireEvent.change(screen.getByLabelText('Answer 1'), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit answer' }))

    expect(await screen.findByText('Not quite.')).toBeInTheDocument()
  })

  it('loads a new problem after submitting', async () => {
    vi.mocked(nextPracticeProblem)
      .mockResolvedValueOnce(sampleProblem)
      .mockResolvedValueOnce({ ...sampleProblem, equation: '3*x = 9' })
    vi.mocked(submitPracticeAnswer).mockResolvedValue({
      status: 'verified_correct',
      detail: 'correct',
      correct: true,
      mastery_estimate: 0.65,
    })

    await startPractice()
    fireEvent.change(screen.getByLabelText('Answer 1'), { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit answer' }))
    await screen.findByText('Correct!')

    fireEvent.click(screen.getByRole('button', { name: 'Next problem' }))

    expect(await screen.findByText('3*x = 9')).toBeInTheDocument()
  })

  it('does not submit while the answer field is empty', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)

    await startPractice()

    expect(screen.getByRole('button', { name: 'Submit answer' })).toBeDisabled()
  })

  it('requests hints via the existing check-work endpoint, scoped to this equation', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)
    vi.mocked(checkWork).mockResolvedValue({
      status: 'all_steps_equivalent',
      steps: [],
      first_invalid_step_index: null,
      first_non_equivalent_step_index: null,
      chain_preserves_solution_set: true,
      deterministic: true,
      detail: '',
      final_answer: null,
      hints: ['Conceptual hint'],
    })

    await startPractice()
    fireEvent.click(screen.getByRole('button', { name: 'Get a hint' }))

    expect(await screen.findByText('Conceptual hint')).toBeInTheDocument()
    expect(checkWork).toHaveBeenCalledWith({
      equation: sampleProblem.equation,
      variable: 'x',
      steps: [],
      hint_level: 1,
    })
  })

  it('persists the learner id across remounts via localStorage', async () => {
    vi.mocked(nextPracticeProblem).mockResolvedValue(sampleProblem)

    const { unmount } = render(<PracticeForm />)
    await screen.findByText(/your mastery/i)
    const firstId = vi.mocked(masterySummary).mock.calls[0][0].learner_id
    unmount()

    render(<PracticeForm />)
    await screen.findByText(/your mastery/i)
    const secondId = vi.mocked(masterySummary).mock.calls[1][0].learner_id

    expect(secondId).toBe(firstId)
  })
})
