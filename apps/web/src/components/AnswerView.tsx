import type { AnswerReason, AnswerResponse } from '../api/types'
import { FeedbackControls } from './FeedbackControls'

export type AskResult =
  | { kind: 'success'; response: AnswerResponse }
  | { kind: 'network-error'; message: string }

const REASON_LABEL: Record<AnswerReason, string> = {
  off_topic: "This doesn't look like a mathematics question.",
  prompt_injection: 'This request could not be processed.',
  insufficient_evidence: "There isn't enough information to answer this responsibly.",
  invalid_output: 'The generated explanation did not pass validation, so it was discarded.',
  retrieval_error: 'The knowledge base is temporarily unavailable.',
  judge_error: 'The evidence-checking service is temporarily unavailable.',
  search_error: 'The web search service is temporarily unavailable.',
  generation_error: 'The explanation service is temporarily unavailable.',
}

function fallbackMessage(status: AnswerResponse['status']): string {
  switch (status) {
    case 'refused':
      return 'This request was not processed.'
    case 'abstained':
      return "There isn't enough information to answer this responsibly."
    case 'unavailable':
      return 'This service is temporarily unavailable.'
    default:
      return ''
  }
}

export function AnswerView({ result }: { result: AskResult }) {
  if (result.kind === 'network-error') {
    return (
      <div className="answer-box answer-box-unavailable">
        <p className="answer-status-label">Connection problem</p>
        <p>{result.message}</p>
      </div>
    )
  }

  const { response } = result

  if (response.status === 'refused' || response.status === 'abstained' || response.status === 'unavailable') {
    const message =
      (response.reason ? REASON_LABEL[response.reason] : null) ??
      response.detail ??
      fallbackMessage(response.status)
    const boxClass =
      response.status === 'unavailable' ? 'answer-box answer-box-unavailable' : 'answer-box answer-box-declined'
    return (
      <div className={boxClass}>
        <p className="answer-status-label">{STATUS_LABEL[response.status]}</p>
        <p>{message}</p>
      </div>
    )
  }

  return (
    <div className="answer-box answer-box-grounded">
      <p className="answer-status-label">{STATUS_LABEL[response.status]}</p>
      <p className="answer-text">{response.answer}</p>

      {response.citations.length > 0 && (
        <div className="citations">
          <p className="citations-label">Sources</p>
          <ul className="citation-list">
            {response.citations.map((citation) => (
              <li key={citation.source_id}>
                <a href={citation.url} target="_blank" rel="noreferrer">
                  {citation.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {response.answer_id !== null && <FeedbackControls key={response.answer_id} answerId={response.answer_id} />}
    </div>
  )
}

const STATUS_LABEL: Record<AnswerResponse['status'], string> = {
  kb: 'From the knowledge base',
  web: 'From a web search',
  refused: 'Not processed',
  abstained: 'Unable to answer',
  unavailable: 'Service unavailable',
}
