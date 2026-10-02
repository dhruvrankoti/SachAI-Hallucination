import { lazy, Suspense, useEffect, useState } from "react"
import Ask from "@/pages/Ask"
import { warmup } from "@/lib/api"

const Verify = lazy(() => import("@/pages/Verify"))
const About = lazy(() => import("@/pages/About"))

const ROUTES = { "": ["Ask", Ask], verify: ["Verify", Verify], about: ["How it works", About] } as const
type Route = keyof typeof ROUTES
const current = () => (location.hash.slice(2) in ROUTES ? location.hash.slice(2) : "") as Route

export default function App() {
  const [route, setRoute] = useState<Route>(current)
  useEffect(() => {
    warmup() // wake the GPU service while the user types
    const on = () => setRoute(current())
    addEventListener("hashchange", on)
    return () => removeEventListener("hashchange", on)
  }, [])
  const Page = ROUTES[route][1]

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-10 bg-bg/80 backdrop-blur border-b border-line">
        <nav className="max-w-3xl mx-auto px-4 h-14 flex items-center justify-between">
          <a href="#/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="h-2.5 w-2.5 rounded-full bg-accent" /> SachAI
          </a>
          <div className="flex gap-1 text-sm">
            {(Object.keys(ROUTES) as Route[]).map((r) => (
              <a key={r} href={`#/${r}`} className={`px-3 py-1.5 rounded-lg transition ${route === r ? "text-fg bg-line/60" : "text-muted hover:text-fg"}`}>{ROUTES[r][0]}</a>
            ))}
          </div>
        </nav>
      </header>
      <main className="flex-1 w-full max-w-3xl mx-auto px-4 py-8">
        <Suspense fallback={null}><Page /></Suspense>
      </main>
      <footer className="text-center text-xs text-muted py-6">Grounded answers, verified by a model trained to catch hallucinations.</footer>
    </div>
  )
}
