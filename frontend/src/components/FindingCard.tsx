import { useState } from 'react'
import { api } from '../api'
import type { EvidenceRef, FindingRecord, ReviewAction, Verdict } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
  finding: FindingRecord
  selected: boolean
  onSelect: () => void
  onViewEvidence: (e: EvidenceRef) => void
  onReviewed: (f: FindingRecord) => void
}

export default function FindingCard({ finding: f, selected, onSelect, onViewEvidence, onReviewed }: Props) {
  const [editing, setEditing] = useState(false)
  const [verdict, setVerdict] = useState<Verdict>(f.final_verdict)
  const [correction, setCorrection] = useState(f.final_correction ?? '')
  const [notes, setNotes] = useState(f.review?.notes ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showChecks, setShowChecks] = useState(false)

  async function submit(action: ReviewAction) {
    setBusy(true)
    setError(null)
    try {
      const review = await api.review(f.finding_id, {
        action,
        notes: notes.trim() || null,
        ...(action === 'edit'
          ? { edited_verdict: verdict, edited_correction: correction.trim() || null }
          : {}),
      })
      onReviewed({
        ...f,
        review,
        final_verdict: action === 'edit' ? verdict : f.verdict,
        final_correction: action === 'edit' ? correction.trim() || f.correction : f.correction,
      })
      setEditing(false)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <article
      id={`finding-${f.claim_id}`}
      className={`card finding finding-${f.final_verdict}${selected ? ' selected' : ''}`}
      onClick={onSelect}
    >
      <div className="finding-head">
        <StatusBadge status={f.final_verdict} />
        {f.error_type && <span className="error-type">{f.error_type.replace('_', ' ')}</span>}
        <span className="muted small">
          {f.verifier} · confidence {Math.round(f.confidence * 100)}%
        </span>
      </div>

      <blockquote>{f.claim_text}</blockquote>
      <p className="rationale">{f.rationale}</p>

      {f.final_correction && (
        <div className="correction">
          <span className="label">Suggested correction</span>
          <p>{f.final_correction}</p>
        </div>
      )}

      {f.evidence.length > 0 && (
        <div className="evidence">
          <span className="label">Evidence</span>
          {f.evidence.map((e) => (
            <button
              key={e.passage_id}
              className="evidence-item"
              onClick={(ev) => {
                ev.stopPropagation()
                onViewEvidence(e)
              }}
            >
              <span className="doc">
                {e.document}
                {e.page ? `, p. ${e.page}` : ''}
              </span>
              <span className="quote">{e.quote}</span>
            </button>
          ))}
        </div>
      )}

      <button
        className="link"
        onClick={(ev) => {
          ev.stopPropagation()
          setShowChecks((s) => !s)
        }}
      >
        {showChecks ? 'Hide' : 'Show'} verifier checks ({f.checks.length})
      </button>
      {showChecks && (
        <ul className="checks">
          {f.checks.map((c, i) => (
            <li key={i}>
              <StatusBadge status={c.verdict} /> <strong>{c.kind}</strong>
              {!c.decisive && <span className="muted small"> (informational)</span>}
              <div className="small">{c.facts.join(' ')}</div>
            </li>
          ))}
        </ul>
      )}

      <div className="review" onClick={(ev) => ev.stopPropagation()}>
        {f.review && (
          <div className="review-status small">
            Reviewed: <strong>{f.review.action}</strong> by {f.review.reviewer}
            {f.review.notes ? ` · “${f.review.notes}”` : ''}
          </div>
        )}
        {editing && (
          <div className="edit-form">
            <label className="small">Verdict</label>
            <select value={verdict} onChange={(e) => setVerdict(e.target.value as Verdict)}>
              <option value="invalid">invalid</option>
              <option value="uncertain">uncertain</option>
              <option value="valid">valid</option>
            </select>
            <label className="small">Correction</label>
            <textarea rows={3} value={correction} onChange={(e) => setCorrection(e.target.value)} />
          </div>
        )}
        <textarea
          className="notes"
          rows={1}
          placeholder="Reviewer notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
        {error && <div className="error small">{error}</div>}
        <div className="review-actions">
          {editing ? (
            <>
              <button className="primary" disabled={busy} onClick={() => submit('edit')}>
                Save edit
              </button>
              <button disabled={busy} onClick={() => setEditing(false)}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button className="accept" disabled={busy} onClick={() => submit('accept')}>
                Accept
              </button>
              <button className="reject" disabled={busy} onClick={() => submit('reject')}>
                Reject
              </button>
              <button disabled={busy} onClick={() => setEditing(true)}>
                Edit
              </button>
            </>
          )}
        </div>
      </div>
    </article>
  )
}
