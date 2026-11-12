export type Verdict = 'valid' | 'invalid' | 'uncertain'
export type ReviewAction = 'accept' | 'reject' | 'edit'

export interface AuditRun {
  id: number
  status: 'pending' | 'running' | 'completed' | 'failed'
  error: string | null
  model: string
  prompt_version: string
  config_hash: string
  created_at: string
  completed_at: string | null
  latency_s: number | null
  cost_usd: number | null
  report_name?: string | null
}

export interface ClaimSummary {
  id: number
  text: string
  span: [number, number]
  claim_type: string
  checkable: boolean
  finding_id: number | null
  verdict: Verdict | null
}

export interface AuditDetail {
  audit: AuditRun
  report: { id: number; name: string; text: string }
  sources: { id: number; name: string; content_type: string; passages: number }[]
  claims: ClaimSummary[]
}

export interface EvidenceRef {
  passage_id: number
  document: string | null
  document_id: number | null
  page: number | null
  char_start: number | null
  char_end: number | null
  quote: string
}

export interface ReviewRecord {
  id: number
  action: ReviewAction
  edited_verdict: Verdict | null
  edited_correction: string | null
  notes: string | null
  reviewer: string
  created_at: string
}

export interface CheckRecord {
  kind: string
  verdict: Verdict
  decisive: boolean
  error_type: string | null
  facts: string[]
  details: Record<string, unknown>
}

export interface FindingRecord {
  finding_id: number
  claim_id: number
  claim_text: string
  claim_type: string
  report_span: [number, number]
  cited_source: string | null
  verdict: Verdict
  error_type: string | null
  rationale: string
  correction: string | null
  confidence: number
  verifier: string
  evidence: EvidenceRef[]
  checks: CheckRecord[]
  review: ReviewRecord | null
  final_verdict: Verdict
  final_correction: string | null
}

export interface ReviewInput {
  action: ReviewAction
  edited_verdict?: Verdict | null
  edited_correction?: string | null
  notes?: string | null
  reviewer?: string | null
}

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  config: () => request<{ default_model: string }>('/config'),
  listAudits: () => request<AuditRun[]>('/audits'),
  getAudit: (id: number) => request<AuditDetail>(`/audits/${id}`),
  getFindings: (id: number) => request<FindingRecord[]>(`/audits/${id}/findings`),
  getDocument: (id: number) =>
    request<{ id: number; name: string; kind: string; text: string }>(`/documents/${id}`),
  createAudit: (form: FormData) =>
    request<{ id: number; status: string }>('/audits', { method: 'POST', body: form }),
  review: (findingId: number, body: ReviewInput) =>
    request<ReviewRecord>(`/findings/${findingId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  exportUrl: (id: number, format: 'md' | 'html' | 'json' | 'jsonl') =>
    `${BASE}/audits/${id}/export?format=${format}`,
}
