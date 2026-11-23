import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, askQuestion, submitFeedback } from './client'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('askQuestion', () => {
  it('posts the question and returns the parsed response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'kb',
        answer: 'x = 2',
        citations: [],
        reason: null,
        detail: null,
        answer_id: 'abc-123',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await askQuestion('What is x in 2x = 4?')

    expect(result.status).toBe('kb')
    expect(result.answer_id).toBe('abc-123')
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/v1/answer')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ question: 'What is x in 2x = 4?' })
  })

  it('throws ApiError with the http status on a non-2xx response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'nope' }, 500)))

    await expect(askQuestion('anything')).rejects.toMatchObject({
      httpStatus: 500,
    })
  })

  it('throws ApiError with no http status when the network request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new TypeError('network down')),
    )

    const error = await askQuestion('anything').catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.httpStatus).toBeUndefined()
  })

  it('re-throws an AbortError instead of wrapping it', async () => {
    const abortError = new DOMException('aborted', 'AbortError')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abortError))

    await expect(askQuestion('anything')).rejects.toBe(abortError)
  })
})

describe('submitFeedback', () => {
  it('posts the feedback payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        feedback_id: 'f-1',
        answer_id: 'abc-123',
        rating: 'up',
        correction: null,
        answer_status: 'kb',
        created_at: '2026-01-01T00:00:00Z',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await submitFeedback({ answer_id: 'abc-123', rating: 'up', correction: null })

    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/v1/feedback')
    expect(JSON.parse(init.body)).toEqual({
      answer_id: 'abc-123',
      rating: 'up',
      correction: null,
    })
  })
})
