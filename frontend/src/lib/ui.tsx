import { CheckCircle2, AlertTriangle, Info } from "lucide-react"
import type { Span } from "./api"

/** Answer text with hallucinated spans underlined. */
export function Highlighted({ text, spans = [] }: { text: string; spans?: Span[] }) {
  const parts: React.ReactNode[] = []
  let i = 0
  for (const s of spans) {
    if (s.start < i) continue
    parts.push(text.slice(i, s.start))
    parts.push(
      <mark key={s.start} title={`${Math.round(s.prob * 100)}% likely unsupported${s.by ? ` (${s.by})` : ""}`}
        className="bg-bad/15 text-fg underline decoration-bad decoration-2 underline-offset-4 rounded-sm px-0.5">
        {text.slice(s.start, s.end)}
      </mark>,
    )
    i = s.end
  }
  parts.push(text.slice(i))
  return <p className="whitespace-pre-wrap leading-7">{parts}</p>
}

export function Badge({ ok, neutral, children }: { ok: boolean; neutral?: boolean; children: React.ReactNode }) {
  const Icon = neutral ? Info : ok ? CheckCircle2 : AlertTriangle
  const tone = neutral ? "bg-line/60 text-muted" : ok ? "bg-ok/10 text-ok" : "bg-bad/10 text-bad"
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${tone}`}>
      <Icon size={13} /> {children}
    </span>
  )
}
