import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { AuditDetail, EvidenceRef, FindingRecord, Verdict } from '../api'
import FindingCard from './FindingCard'
import ReportPane from './ReportPane'
import SourceViewer from './SourceViewer'
import StatusBadge from './StatusBadge'

type Filter = 'all' | Verdict

export default function AuditView({ auditId }: { auditId: number }) {
  const [detail, setDetail] = useState<AuditDetail | null>(null)
  const [findings, setFindings] = useState<FindingRecord[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [viewing, setViewing] = useState<EvidenceRef | null>(null)

  const status = detail?.audit.status
  const done = status === 'completed' || status === 'failed'

  const load = useCallback(async () => {
    try {
      const d = await api.getAudit(auditId)
      const f = d.audit.status === 'completed' ? await api.getFindings(auditId) : []
      setFindings(f)
      setDetail(d)
    } catch (err) {
      setError((err as Error).message)
    }
  }, [auditId])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!detail || done) return
    const t = setInterval(load, 1500)
    return () => clearInterval(t)
  }, [detail, done, load])

  const byClaim = useMemo(() => new Map(findings.map((f) => [f.claim_id, f])), [findings])
  const counts = useMemo(() => {
    const c: Record<string, number> = { invalid: 0, uncertain: 0, valid: 0 }
    findings.forEach((f) => (c[f.final_verdict] += 1))
    return c
  }, [findings])
  const shown = findings
    .filter((f) => filter === 'all' || f.final_verdict === filter)
    .sort((a, b) => order(a.final_verdict) - order(b.final_verdict) || a.report_span[0] - b.report_span[0])

  function selectClaim(claimId: number) {
    setSelected(claimId)
    document.getElementById(`finding-${claimId}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  function onReviewed(updated: FindingRecord) {
    setFindings((fs) => fs.map((f) => (f.finding_id === updated.finding_id ? updated : f)))
  }

  if (error) return <div className="card error">{error}</div>
  if (!detail) return <div className="card muted">Loading…</div>

  const { audit } = detail
  return (
    <div className="audit">
      <div className="audit-header card">
        <div>
          <h1>
            Audit #{audit.id}: {detail.report.name}
          </h1>
          <div className="meta">
            <StatusBadge status={audit.status} />
            <span>model {audit.model}</span>
            <span>prompts {audit.prompt_version}</span>
            {audit.latency_s != null && <span>{audit.latency_s.toFixed(1)}s</span>}
            {audit.cost_usd ? <span>${audit.cost_usd.toFixed(4)}</span> : null}
            <span>sources: {detail.sources.map((s) => s.name).join(', ')}</span>
          </div>
          {audit.status === 'failed' && <div className="error">{audit.error}</div>}
        </div>
        {audit.status === 'completed' && (
          <div className="exports">
            {(['md', 'html', 'json', 'jsonl'] as const).map((fmt) => (
              <a key={fmt} className="button" href={api.exportUrl(audit.id, fmt)}>
                {fmt.toUpperCase()}
              </a>
            ))}
          </div>
        )}
      </div>

      {!done && (
        <div className="card running">
          <div className="spinner" /> Extracting and verifying claims…
        </div>
      )}

      {done && (
        <div className="split">
          <ReportPane
            text={detail.report.text}
            claims={detail.claims}
            findings={byClaim}
            selected={selected}
            onSelect={selectClaim}
          />
          <section className="findings">
            <div className="filters">
              {(['all', 'invalid', 'uncertain', 'valid'] as const).map((f) => (
                <button key={f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>
                  {f} {f === 'all' ? findings.length : counts[f]}
                </button>
              ))}
            </div>
            {shown.length === 0 && <p className="muted">No findings in this category.</p>}
            {shown.map((f) => (
              <FindingCard
                key={f.finding_id}
                finding={f}
                selected={selected === f.claim_id}
                onSelect={() => setSelected(f.claim_id)}
                onViewEvidence={setViewing}
                onReviewed={onReviewed}
              />
            ))}
          </section>
        </div>
      )}

      {viewing && <SourceViewer evidence={viewing} onClose={() => setViewing(null)} />}
    </div>
  )
}

function order(v: Verdict): number {
  return { invalid: 0, uncertain: 1, valid: 2 }[v]
}
