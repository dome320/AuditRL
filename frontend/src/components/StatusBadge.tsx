export default function StatusBadge({ status }: { status: string | null }) {
  if (!status) return <span className="badge muted-badge">not checked</span>
  return <span className={`badge badge-${status}`}>{status}</span>
}
