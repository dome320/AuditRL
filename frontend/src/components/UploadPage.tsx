import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../api'
import type { AuditRun } from '../api'
import StatusBadge from './StatusBadge'

const ACCEPT = '.md,.markdown,.txt,.pdf,.html,.htm,.csv,.xlsx'

export default function UploadPage() {
  const [mode, setMode] = useState<'file' | 'paste'>('paste')
  const [reportFile, setReportFile] = useState<File | null>(null)
  const [reportText, setReportText] = useState('')
  const [sources, setSources] = useState<File[]>([])
  const [model, setModel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [audits, setAudits] = useState<AuditRun[]>([])

  useEffect(() => {
    api.config().then((c) => setModel((m) => m || c.default_model)).catch(() => {})
    api.listAudits().then(setAudits).catch(() => {})
  }, [])

  async function submit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    const form = new FormData()
    if (mode === 'file') {
      if (!reportFile) return setError('Choose a report file.')
      form.append('report', reportFile)
    } else {
      if (!reportText.trim()) return setError('Paste the report text.')
      form.append('report_text', reportText)
    }
    if (sources.length === 0) return setError('Attach at least one source document.')
    sources.forEach((s) => form.append('sources', s))
    if (model.trim()) form.append('model', model.trim())
    setSubmitting(true)
    try {
      const { id } = await api.createAudit(form)
      window.location.hash = `#/audits/${id}`
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="upload-layout">
      <form className="card upload" onSubmit={submit}>
        <h1>New audit</h1>
        <p className="muted">
          Submit a research report and the filings or tables it relies on. AuditRL extracts
          checkable claims, verifies figures and arithmetic against the sources, and proposes
          corrections.
        </p>

        <div className="field">
          <div className="label-row">
            <label>Report</label>
            <div className="segmented">
              <button type="button" className={mode === 'paste' ? 'on' : ''} onClick={() => setMode('paste')}>
                Paste
              </button>
              <button type="button" className={mode === 'file' ? 'on' : ''} onClick={() => setMode('file')}>
                Upload
              </button>
            </div>
          </div>
          {mode === 'paste' ? (
            <textarea
              rows={10}
              value={reportText}
              onChange={(e) => setReportText(e.target.value)}
              placeholder="Apple reported total net sales of $383.3 billion in fiscal 2023..."
            />
          ) : (
            <input type="file" accept={ACCEPT} onChange={(e) => setReportFile(e.target.files?.[0] ?? null)} />
          )}
        </div>

        <div className="field">
          <label>Source documents</label>
          <input
            type="file"
            multiple
            accept={ACCEPT}
            onChange={(e) => setSources(Array.from(e.target.files ?? []))}
          />
          {sources.length > 0 && (
            <ul className="file-list">
              {sources.map((s) => (
                <li key={s.name}>{s.name}</li>
              ))}
            </ul>
          )}
        </div>

        <div className="field">
          <label>Model</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="anthropic/claude-sonnet-4-5, openai/gpt-4o-mini, heuristic"
          />
          <span className="hint">Any LiteLLM model string, or "heuristic" to run offline without an API key.</span>
        </div>

        {error && <div className="error">{error}</div>}
        <button className="primary" type="submit" disabled={submitting}>
          {submitting ? 'Submitting…' : 'Run audit'}
        </button>
      </form>

      <section className="card recent">
        <h2>Recent audits</h2>
        {audits.length === 0 ? (
          <p className="muted">No audits yet.</p>
        ) : (
          <ul className="audit-list">
            {audits.map((a) => (
              <li key={a.id}>
                <a href={`#/audits/${a.id}`}>
                  <span>#{a.id} {a.report_name}</span>
                  <StatusBadge status={a.status} />
                </a>
                <span className="muted small">
                  {a.model} · {new Date(a.created_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
