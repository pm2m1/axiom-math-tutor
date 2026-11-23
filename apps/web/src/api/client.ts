import type { AnswerResponse, FeedbackRequest, FeedbackResponse } from './types'

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export const QUESTION_MAX_LENGTH = 500

/** Thrown for network failures and non-2xx responses — distinct from a
 * well-formed `AnswerResponse` with `status: "unavailable"`, which is
 * the backend's own typed way of reporting a dependency failure. This
 * is only for "the request itself didn't complete." `httpStatus` is
 * `undefined` for a pure network failure (no response at all). */
export class ApiError extends Error {
  readonly httpStatus: number | undefined

  constructor(message: string, httpStatus?: number) {
    super(message)
    this.httpStatus = httpStatus
  }
}

async function postJson<TResponse>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<TResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('network request failed')
  }
  if (!response.ok) {
    throw new ApiError(`request failed with status ${response.status}`, response.status)
  }
  return (await response.json()) as TResponse
}

export function askQuestion(question: string, signal?: AbortSignal): Promise<AnswerResponse> {
  return postJson<AnswerResponse>('/api/v1/answer', { question }, signal)
}

export function submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
  return postJson<FeedbackResponse>('/api/v1/feedback', request)
}
