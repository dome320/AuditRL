import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { EvidenceRef } from '../api'

export default function SourceViewer({ evidence, onClose }: { evidence: EvidenceRef; onClose: () => void }) {
  const [doc, setDoc] = useState<{ name: string; text: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const markRef = useRef<HTMLElement>(null)

  useEffect(() => {
    if (evidence.document_id == null) return
    api.getDocument(evidence.document_id).then(setDoc).catch((e) => setError(e.message))
  }, [evidence.document_id])

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: 'center' })
  }, [doc])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const start = evidence.char_start ?? 0
  const end = evidence.char_end ?? 0

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>
            {evidence.document}
            {evidence.page ? `, p. ${evidence.page}` : ''}
          </h2>
          <button onClick={onClose}>Close</button>
        </div>
        {error && <div className="error">{error}</div>}
        {!doc && !error && <p className="muted">Loading…</p>}
        {doc && (
          <pre className="source-text">
            {doc.text.slice(0, start)}
            <mark ref={markRef}>{doc.text.slice(start, end)}</mark>
            {doc.text.slice(end)}
          </pre>
        )}
      </div>
    </div>
  )
}
