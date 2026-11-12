import type { ClaimSummary, FindingRecord } from '../api'

interface Props {
  text: string
  claims: ClaimSummary[]
  findings: Map<number, FindingRecord>
  selected: number | null
  onSelect: (claimId: number) => void
}

type Segment = { text: string; claim?: ClaimSummary }

function segments(text: string, claims: ClaimSummary[]): Segment[] {
  const sorted = [...claims].sort((a, b) => a.span[0] - b.span[0])
  const out: Segment[] = []
  let pos = 0
  for (const c of sorted) {
    const [start, end] = c.span
    if (start < pos) continue
    if (start > pos) out.push({ text: text.slice(pos, start) })
    out.push({ text: text.slice(start, end), claim: c })
    pos = end
  }
  if (pos < text.length) out.push({ text: text.slice(pos) })
  return out
}

export default function ReportPane({ text, claims, findings, selected, onSelect }: Props) {
  return (
    <section className="card report">
      <h2>Report</h2>
      <div className="legend">
        <span className="span-invalid">invalid</span>
        <span className="span-uncertain">uncertain</span>
        <span className="span-valid">valid</span>
        <span className="span-unchecked">not checkable</span>
      </div>
      <div className="report-text">
        {segments(text, claims).map((s, i) => {
          if (!s.claim) return <span key={i}>{s.text}</span>
          const finding = findings.get(s.claim.id)
          const verdict = finding?.final_verdict ?? (s.claim.checkable ? s.claim.verdict : null)
          const cls = `claim span-${verdict ?? 'unchecked'}${selected === s.claim.id ? ' selected' : ''}`
          return (
            <span
              key={i}
              className={cls}
              title={finding ? `${verdict}${finding.error_type ? ` · ${finding.error_type}` : ''}` : s.claim.claim_type}
              onClick={() => finding && onSelect(s.claim!.id)}
            >
              {s.text}
            </span>
          )
        })}
      </div>
    </section>
  )
}
