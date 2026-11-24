import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkWork } from '../api/client'
import type { CheckWorkResponse } from '../api/types'
import { CheckWorkForm } from './CheckWorkForm'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, checkWork: vi.fn() }
})

afterEach(() => {
  vi.mocked(checkWork).mockReset()
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

function baseResponse(overrides: Partial<CheckWorkResponse> = {}): CheckWorkResponse {
  return {
    status: 'all_steps_equivalent',
    steps: [],
    first_invalid_step_index: null,
    first_non_equivalent_step_index: null,
    chain_preserves_solution_set: true,
    deterministic: true,
    detail: 'every step preserves the original real solution set exactly',
    final_answer: null,
    hints: null,
    ...overrides,
  }
}

function setEquation(value: string) {
  fireEvent.change(screen.getByLabelText('Original equation'), { target: { value } })
}

function submit() {
  fireEvent.click(screen.getByRole('button', { name: /check my work|checking/i }))
}

describe('CheckWorkForm', () => {
  it('lets the student add and remove step inputs', () => {
    render(<CheckWorkForm />)

    fireEvent.click(screen.getByRole('button', { name: 'Add step' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add step' }))
    fireEvent.change(screen.getByLabelText('Step 1'), { target: { value: 'first' } })
    fireEvent.change(screen.getByLabelText('Step 2'), { target: { value: 'second' } })
    expect(screen.queryByLabelText('Step 2')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Remove step 1' }))

    // Removing the first row leaves one input, renumbered to "Step 1",
    // now holding what was previously the second row's value.
    expect(screen.queryByLabelText('Step 2')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Step 1')).toHaveValue('second')
  })

  it('submits the equation, steps, and omits solutions by default', async () => {
    vi.mocked(checkWork).mockResolvedValue(baseResponse())

    render(<CheckWorkForm />)
    setEquation('2*x + 3 = 11')
    fireEvent.click(screen.getByRole('button', { name: 'Add step' }))
    fireEvent.change(screen.getByLabelText('Step 1'), { target: { value: '2*x = 8' } })
    submit()

    await screen.findByText(/all steps preserve the solution set/i)
    expect(checkWork).toHaveBeenCalledWith(
      { equation: '2*x + 3 = 11', steps: ['2*x = 8'], solutions: undefined },
      expect.anything(),
    )
  })

  it('sends an explicit empty solutions list when claiming no real solutions', async () => {
    vi.mocked(checkWork).mockResolvedValue(baseResponse())

    render(<CheckWorkForm />)
    setEquation('x^2 + 1 = 0')
    fireEvent.click(screen.getByRole('button', { name: 'This equation has no real solutions' }))
    submit()

    await screen.findByText(/all steps preserve the solution set/i)
    expect(checkWork).toHaveBeenCalledWith(
      { equation: 'x^2 + 1 = 0', steps: [], solutions: [] },
      expect.anything(),
    )
  })

  it('sends listed solution values when solutions are added', async () => {
    vi.mocked(checkWork).mockResolvedValue(baseResponse())

    render(<CheckWorkForm />)
    setEquation('2*x = 4')
    fireEvent.click(screen.getByRole('button', { name: 'Add solution' }))
    fireEvent.change(screen.getByLabelText('Proposed solution 1'), { target: { value: '2' } })
    submit()

    await screen.findByText(/all steps preserve the solution set/i)
    expect(checkWork).toHaveBeenCalledWith(
      { equation: '2*x = 4', steps: [], solutions: ['2'] },
      expect.anything(),
    )
  })

  it('shows a loading state and disables the submit button while in flight', async () => {
    const { promise, resolve } = deferred<CheckWorkResponse>()
    vi.mocked(checkWork).mockReturnValue(promise)

    render(<CheckWorkForm />)
    setEquation('2*x = 4')
    submit()

    const button = screen.getByRole('button', { name: /checking/i })
    expect(button).toBeDisabled()

    resolve(baseResponse())
    await screen.findByText(/all steps preserve the solution set/i)
  })

  it('does not start a second request while one is already in flight', () => {
    const { promise } = deferred<CheckWorkResponse>()
    vi.mocked(checkWork).mockReturnValue(promise)

    render(<CheckWorkForm />)
    setEquation('2*x = 4')
    submit()
    fireEvent.click(screen.getByRole('button', { name: /checking/i }))

    expect(checkWork).toHaveBeenCalledTimes(1)
  })

  it('highlights the first provably incorrect step', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'invalid_step_found',
        first_invalid_step_index: 2,
        first_non_equivalent_step_index: 2,
        chain_preserves_solution_set: false,
        detail: 'a provably incorrect step was found',
        steps: [
          {
            index: 1,
            equation: '2*x = 8',
            category: 'linear',
            transition: 'equivalent',
            detail: 'this step preserves the real solution set',
            misconception: null,
          },
          {
            index: 2,
            equation: 'x = 5',
            category: 'linear',
            transition: 'incompatible',
            detail: 'this step changes the solution set incompatibly with the previous equation',
            misconception: null,
          },
        ],
      }),
    )

    render(<CheckWorkForm />)
    setEquation('2*x + 3 = 11')
    submit()

    expect(await screen.findByText(/first error/i)).toBeInTheDocument()
    expect(screen.getByText('Incompatible — provably incorrect')).toBeInTheDocument()
    expect(screen.getByText('Equivalent')).toBeInTheDocument()
  })

  it('flags a partial-overlap step as the first error, not as a mere warning', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'invalid_step_found',
        first_invalid_step_index: 1,
        first_non_equivalent_step_index: 1,
        chain_preserves_solution_set: false,
        detail: 'a provably incorrect step was found',
        steps: [
          {
            index: 1,
            equation: '(x-2)*(x-3) = 0',
            category: 'quadratic',
            transition: 'partial_overlap',
            detail:
              'this step drops at least one previously valid solution while introducing a different, unverified one',
            misconception: 'general_error',
          },
        ],
      }),
    )

    render(<CheckWorkForm />)
    setEquation('(x-1)*(x-2) = 0')
    submit()

    expect(await screen.findByText(/first error/i)).toBeInTheDocument()
    expect(screen.getByText('Partial overlap — provably incorrect')).toBeInTheDocument()
  })

  it('shows non-equivalent warnings for narrowing and broadening without flagging them as errors', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'valid_but_incomplete',
        first_non_equivalent_step_index: 1,
        chain_preserves_solution_set: false,
        detail: 'no provably incorrect step was found, but at least one step changed the solution set',
        steps: [
          {
            index: 1,
            equation: 'x = 1',
            category: 'linear',
            transition: 'narrowing',
            detail: 'this step removes one or more possible solutions',
            misconception: 'lost_solution_branch',
          },
        ],
      }),
    )

    render(<CheckWorkForm />)
    setEquation('x^2 = 1')
    submit()

    expect(await screen.findByText('Narrows the solution set')).toBeInTheDocument()
    expect(screen.queryByText(/first error/i)).not.toBeInTheDocument()
  })

  it('renders the independent final-answer result', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        final_answer: {
          status: 'verified_correct',
          category: 'linear',
          domain: 'real',
          deterministic: true,
          reason: null,
          detail: 'the proposed solution set matches the exact real solution set',
          complete: true,
          candidates: [{ value: '4', valid: true }],
        },
      }),
    )

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    fireEvent.click(screen.getByRole('button', { name: 'Add solution' }))
    fireEvent.change(screen.getByLabelText('Proposed solution 1'), { target: { value: '4' } })
    submit()

    expect(await screen.findByText(/final answer: correct and complete/i)).toBeInTheDocument()
    expect(screen.getByText('4: valid')).toBeInTheDocument()
  })

  it('renders an unsupported response', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'unsupported',
        detail: 'the original equation is outside the supported verification grammar',
      }),
    )

    render(<CheckWorkForm />)
    setEquation('x^3 - x = 0')
    submit()

    expect(await screen.findByText(/outside the supported grammar/i)).toBeInTheDocument()
  })

  it('renders an indeterminate/incomplete-verification response', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'verification_incomplete',
        chain_preserves_solution_set: null,
        detail: 'at least one step could not be verified',
        steps: [
          {
            index: 1,
            equation: 'x + y = 1',
            category: 'unsupported',
            transition: 'unsupported',
            detail: "could not parse this step: unknown symbol 'y'",
            misconception: null,
          },
        ],
      }),
    )

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    submit()

    expect(await screen.findByText(/verification was incomplete/i)).toBeInTheDocument()
    expect(screen.getByText('Unsupported')).toBeInTheDocument()
  })

  it('shows a network-error message when the request fails', async () => {
    vi.mocked(checkWork).mockRejectedValue(new Error('network down'))

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    submit()

    expect(await screen.findByText(/could not reach the server/i)).toBeInTheDocument()
  })

  it('shows a misconception label for a step that has one', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({
        status: 'invalid_step_found',
        first_invalid_step_index: 1,
        steps: [
          {
            index: 1,
            equation: 'x = -4',
            category: 'linear',
            transition: 'incompatible',
            detail: 'this step changes the solution set incompatibly with the previous equation',
            misconception: 'sign_error',
          },
        ],
      }),
    )

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    submit()

    expect(await screen.findByText(/sign error/i)).toBeInTheDocument()
  })

  it('requests level-1 hints on the first click and progresses on subsequent clicks', async () => {
    vi.mocked(checkWork)
      .mockResolvedValueOnce(baseResponse({ hints: ['Conceptual hint'] }))
      .mockResolvedValueOnce(baseResponse({ hints: ['Conceptual hint', 'Directional hint'] }))

    render(<CheckWorkForm />)
    setEquation('2*x = 8')

    fireEvent.click(screen.getByRole('button', { name: 'Get a hint' }))
    expect(await screen.findByText('Conceptual hint')).toBeInTheDocument()
    expect(checkWork).toHaveBeenCalledWith({ equation: '2*x = 8', steps: [], hint_level: 1 })

    fireEvent.click(screen.getByRole('button', { name: 'Get another hint' }))
    expect(await screen.findByText('Directional hint')).toBeInTheDocument()
    expect(checkWork).toHaveBeenLastCalledWith({ equation: '2*x = 8', steps: [], hint_level: 2 })
  })

  it('disables the hint button once level 4 is reached', async () => {
    vi.mocked(checkWork).mockResolvedValue(
      baseResponse({ hints: ['L1', 'L2', 'L3', 'L4: the answer'] }),
    )

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    fireEvent.click(screen.getByRole('button', { name: 'Get a hint' }))

    expect(await screen.findByText('L4: the answer')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Get another hint' })).toBeDisabled()
  })

  it('clears hints when the equation is edited', async () => {
    vi.mocked(checkWork).mockResolvedValue(baseResponse({ hints: ['Conceptual hint'] }))

    render(<CheckWorkForm />)
    setEquation('2*x = 8')
    fireEvent.click(screen.getByRole('button', { name: 'Get a hint' }))
    expect(await screen.findByText('Conceptual hint')).toBeInTheDocument()

    setEquation('3*x = 9')
    expect(screen.queryByText('Conceptual hint')).not.toBeInTheDocument()
  })
})
