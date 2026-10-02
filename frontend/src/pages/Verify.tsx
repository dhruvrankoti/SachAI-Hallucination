import { useState } from "react"
import { Loader2, ScanSearch } from "lucide-react"
import { verify, type Verdict } from "@/lib/api"
import { Badge, Highlighted } from "@/lib/ui"

const SAMPLE = {
  context: "The Eiffel Tower was completed in 1889 for the Exposition Universelle in Paris. It is 330 metres tall and was designed by the engineering firm of Gustave Eiffel.",
  answer: "The Eiffel Tower was completed in 1889 and stands 330 metres tall. It was designed by Gustave Eiffel's firm and was originally painted gold for the 1900 Olympics.",
}

export default function Verify() {
  const [context, setContext] = useState("")
  const [answer, setAnswer] = useState("")
  const [checked, setChecked] = useState("")
  const [res, setRes] = useState<Verdict>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  async function run() {
    setBusy(true); setError(""); setRes(undefined)
    try { setRes(await verify(context, answer)); setChecked(answer) } catch (e) { setError((e as Error).message) }
    setBusy(false)
  }

  return (
    <div className="space-y-6 fade">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Verify any answer</h1>
        <p className="text-muted text-sm">Paste a source and an AI-generated answer. SachAI highlights every claim the source doesn't support.</p>
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <label className="space-y-2">
          <span className="label">Source / context</span>
          <textarea className="input h-52 resize-none" value={context} onChange={(e) => setContext(e.target.value)} placeholder="The ground-truth text…" />
        </label>
        <label className="space-y-2">
          <span className="label">AI answer</span>
          <textarea className="input h-52 resize-none" value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder="The response to check…" />
        </label>
      </div>
      <div className="flex gap-2">
        <button className="btn-primary" onClick={run} disabled={busy || !context.trim() || !answer.trim()}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <ScanSearch size={16} />} Check
        </button>
        <button className="btn-ghost" onClick={() => { setContext(SAMPLE.context); setAnswer(SAMPLE.answer) }}>Try a sample</button>
      </div>
      {error && <div className="card border-bad/30 bg-bad/5 text-bad text-sm p-4">{error}</div>}
      {res && (
        <div className="card p-6 space-y-4 fade">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <Badge ok={!res.hallucinated}>{res.hallucinated ? `${res.spans.length} unsupported claim${res.spans.length === 1 ? "" : "s"}` : "Fully supported"}</Badge>
            <span className="text-xs text-muted" title="Detector: highest hallucination probability on any word. Jev: how well the least-supported sentence is backed by the source.">
              Detector risk {Math.round(res.score * 100)}%{res.jev != null && ` · Jev: ${Math.round(res.jev * 100)}% supported`}
            </span>
          </div>
          <Highlighted text={checked} spans={res.spans} />
        </div>
      )}
    </div>
  )
}
