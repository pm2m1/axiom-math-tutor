import { useEffect, useState } from 'react'

type ApiState = 'checking' | 'online' | 'offline'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const POLL_INTERVAL_MS = 15_000

const LABEL: Record<ApiState, string> = {
  checking: 'checking api',
  online: 'api online',
  offline: 'api offline',
}

/**
 * Small development affordance so it's obvious whether the FastAPI backend
 * is reachable. Not part of the product surface — remove or hide behind a
 * dev flag once there's a real status/observability story.
 */
export function StatusIndicator() {
  const [state, setState] = useState<ApiState>('checking')

  useEffect(() => {
    let cancelled = false

    async function check() {
      try {
        const response = await fetch(`${API_BASE}/api/v1/status`)
        if (!cancelled) setState(response.ok ? 'online' : 'offline')
      } catch {
        if (!cancelled) setState('offline')
      }
    }

    check()
    const interval = setInterval(check, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <span className={`status status-${state}`} title={`API base: ${API_BASE}`}>
      <span className="status-dot" aria-hidden="true" />
      {LABEL[state]}
    </span>
  )
}
