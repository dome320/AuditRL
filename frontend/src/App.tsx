import { useEffect, useState } from 'react'
import AuditView from './components/AuditView'
import UploadPage from './components/UploadPage'

function parseRoute(hash: string): { page: 'home' } | { page: 'audit'; id: number } {
  const m = hash.match(/^#\/audits\/(\d+)/)
  return m ? { page: 'audit', id: Number(m[1]) } : { page: 'home' }
}

export default function App() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash))

  useEffect(() => {
    const onHash = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand">
          AuditRL
        </a>
        <span className="tagline">Source-grounded audits of financial research reports</span>
      </header>
      <main>
        {route.page === 'audit' ? <AuditView key={route.id} auditId={route.id} /> : <UploadPage />}
      </main>
    </div>
  )
}
