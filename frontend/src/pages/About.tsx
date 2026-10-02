import { useEffect, useState } from "react"
import { getStats, type Stats } from "@/lib/api"

const STEPS = [
  ["Retrieve", "Your uploaded documents are searched with Postgres full-text search. Without documents, Wikipedia pages are fetched in parallel (LLM-suggested titles plus search). Jev re-ranks the candidates and keeps the top 5 passages."],
  ["Generate", "The LLM you pick (via Groq or OpenRouter) writes an answer constrained to those passages, streamed token by token."],
  ["Verify", "Jev checks every sentence of the answer against the passages in one call (~0.3 s), plus a numeric check. Unsupported sentences go back to the model to rewrite, up to 3 retries."],
  ["Deep check", "The trained ModernBERT detector (fine-tuned on RAGTruth) then audits the final answer in the background and highlights any phrase it flags."],
]

// training/results/v1.json: trained detector on the RAGTruth official test split (2,700 responses)
const BENCH = [["Example F1", "0.787"], ["Precision", "0.783"], ["Recall", "0.791"], ["Token F1", "0.579"]]

export default function About() {
  const [s, setS] = useState<Stats>()
  useEffect(() => { getStats().then(setS).catch(() => {}) }, [])
  const pct = (n: number) => (s?.questions ? `${Math.round((n / s.questions) * 100)}%` : "—")

  return (
    <div className="space-y-10 fade max-w-2xl">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">How SachAI works</h1>
        <p className="text-muted">A retrieve, generate, detect, correct loop that keeps answers tied to their sources.</p>
      </div>
      <ol className="space-y-6">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="flex gap-4">
            <span className="h-7 w-7 shrink-0 rounded-full border border-line grid place-items-center text-xs font-medium">{i + 1}</span>
            <div><div className="font-medium">{t}</div><p className="text-muted text-sm leading-6">{d}</p></div>
          </li>
        ))}
      </ol>

      <section className="space-y-3">
        <div className="label">Detector benchmark · RAGTruth test set (2,700 responses)</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {BENCH.map(([k, v]) => <Tile key={k} k={k} v={v} />)}
        </div>
      </section>

      <section className="space-y-3">
        <div className="label">Live usage · from the database</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Tile k="Questions" v={s ? String(s.questions) : "—"} />
          <Tile k="Verified" v={s ? pct(s.verified) : "—"} />
          <Tile k="Needed a retry" v={s ? pct(s.needed_retry) : "—"} />
          <Tile k="Avg latency" v={s?.avg_latency_ms != null ? `${(s.avg_latency_ms / 1000).toFixed(1)}s` : "—"} />
        </div>
      </section>
    </div>
  )
}

function Tile({ k, v }: { k: string; v: string }) {
  return (
    <div className="card p-4">
      <div className="text-xl font-semibold tabular-nums">{v}</div>
      <div className="text-xs text-muted mt-1">{k}</div>
    </div>
  )
}
