import { useEffect, useRef, useState } from "react"
import { ArrowUp, FileText, Globe, Loader2, Paperclip, RotateCcw, X } from "lucide-react"
import { ask, clearDocs, listDocs, listModels, upload, type Source, type Span, type Verdict } from "@/lib/api"
import { DEFAULT_MODEL, MODELS, ModelPicker } from "@/lib/ModelPicker"
import { Badge, Highlighted } from "@/lib/ui"

interface Attempt { text: string; verdict?: Verdict }

const MODEL_KEY = "sachai.model.v2"  // v2: earlier saved picks reset once to the new default

const EXAMPLES = ["Who discovered penicillin and when?", "What is the tallest mountain in Africa?", "How does CRISPR gene editing work?"]

export default function Ask() {
  const [q, setQ] = useState("")
  const [asked, setAsked] = useState("")
  const [busy, setBusy] = useState(false)
  const [docs, setDocs] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [sources, setSources] = useState<Source[]>([])
  const [mode, setMode] = useState<"docs" | "web" | "direct">()
  const [attempts, setAttempts] = useState<Attempt[]>([])
  const [final, setFinal] = useState<{ answer: string; verified: boolean; best: number; refused?: boolean; direct?: boolean }>()
  const [error, setError] = useState("")
  const [available, setAvailable] = useState<string[]>()
  const [audit, setAudit] = useState<{ hallucinated?: boolean; spans?: Span[]; unavailable?: boolean } | "pending">()
  const [model, setModel] = useState(() => {  // synchronous: the picker shows instantly, no API wait
    let saved = ""
    try { saved = localStorage.getItem(MODEL_KEY) ?? "" } catch { /* storage unavailable */ }
    return MODELS.some((m) => m.id === saved) ? saved : DEFAULT_MODEL
  })
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController>(null)

  useEffect(() => {
    listDocs().then((r) => setDocs(r.docs)).catch(() => {})
    // the backend says which providers are configured; hide the rest
    listModels().then((r) => {
      const ids = r.models.map((m) => m.id)
      setAvailable(ids)
      setModel((m) => (ids.includes(m) ? m : r.default ?? m))
    }).catch(() => {})
  }, [])

  function pickModel(id: string) {
    setModel(id)
    try { localStorage.setItem(MODEL_KEY, id) } catch { /* storage unavailable */ }
  }

  async function submit(question = q) {
    question = question.trim()
    if (!question || busy) return
    abortRef.current?.abort()
    const ctrl = (abortRef.current = new AbortController())
    setAsked(question); setQ(""); setBusy(true); setError("")
    setSources([]); setAttempts([]); setFinal(undefined); setMode(undefined); setAudit(undefined)
    try {
      await ask(question, model, (e) => {
        if (e.type === "sources") { setSources(e.sources); setMode(e.mode) }
        else if (e.type === "attempt") setAttempts((a) => [...a, { text: "" }])
        else if (e.type === "token") setAttempts((a) => a.map((x, i) => (i === e.n ? { ...x, text: x.text + e.text } : x)))
        else if (e.type === "verdict") setAttempts((a) => a.map((x, i) => (i === e.n ? { ...x, verdict: e } : x)))
        else if (e.type === "final") { setFinal(e); if (!e.refused && !e.direct) setAudit("pending") }
        else if (e.type === "audit") setAudit(e)
        else if (e.type === "error") setError(e.message)
      }, ctrl.signal)
    } catch (err) {
      if (!ctrl.signal.aborted) setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function onFiles(files: FileList | null) {
    if (!files?.length) return
    setUploading(true); setError("")
    try { setDocs((await upload([...files])).docs) } catch (e) { setError((e as Error).message) }
    setUploading(false)
    if (fileRef.current) fileRef.current.value = ""
  }

  const best = final ? attempts[final.best] : undefined
  const current = attempts[attempts.length - 1]

  return (
    <div className="space-y-8">
      {!asked && (
        <div className="pt-10 sm:pt-20 text-center space-y-3 fade">
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">Answers you can trust.</h1>
          <p className="text-muted max-w-md mx-auto">Every answer is fact-checked against its sources. If anything is made up, SachAI catches it and rewrites it.</p>
        </div>
      )}

      <form onSubmit={(e) => { e.preventDefault(); submit() }} className="card p-2 shadow-sm focus-within:border-fg/30 transition">
        <textarea value={q} onChange={(e) => setQ(e.target.value)} rows={2} placeholder={docs.length ? "Ask about your documents…" : "Ask anything — grounded in Wikipedia, or upload your own docs"}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit() } }}
          className="w-full resize-none bg-transparent px-3 py-2 text-[15px] outline-none placeholder:text-muted/70" />
        <div className="flex items-center justify-between gap-2 px-1">
          <div className="flex items-center gap-1 flex-wrap min-w-0">
            <input ref={fileRef} type="file" multiple accept=".pdf,.txt,.md" hidden onChange={(e) => onFiles(e.target.files)} />
            <button type="button" className="btn-ghost h-8 px-2.5" onClick={() => fileRef.current?.click()} disabled={uploading}>
              {uploading ? <Loader2 size={15} className="animate-spin" /> : <Paperclip size={15} />} <span className="hidden sm:inline">Upload</span>
            </button>
            <ModelPicker value={model} onChange={pickModel} available={available} />
            {docs.map((d) => (
              <span key={d} className="inline-flex items-center gap-1 text-xs bg-line/60 rounded-lg px-2 py-1 max-w-[160px]">
                <FileText size={12} className="shrink-0" /><span className="truncate">{d}</span>
              </span>
            ))}
            {docs.length > 0 && (
              <button type="button" title="Remove documents" className="btn-ghost h-7 w-7 px-0" onClick={() => clearDocs().then((r) => setDocs(r.docs))}><X size={14} /></button>
            )}
          </div>
          <button className="btn-primary h-9 w-9 px-0 rounded-full shrink-0" disabled={!q.trim() || busy} aria-label="Ask">
            {busy ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={16} />}
          </button>
        </div>
      </form>

      {!asked && (
        <div className="flex flex-wrap justify-center gap-2 fade">
          {EXAMPLES.map((e) => <button key={e} onClick={() => submit(e)} className="text-sm text-muted border border-line rounded-full px-3.5 py-1.5 hover:text-fg hover:border-fg/20 transition">{e}</button>)}
        </div>
      )}

      {asked && (
        <section className="space-y-5 fade">
          <h2 className="text-xl font-semibold tracking-tight">{asked}</h2>

          {error && <div className="card border-bad/30 bg-bad/5 text-bad text-sm p-4">{error}</div>}

          {!mode && busy && <Status text="Searching sources…" />}

          {/* Final / live answer */}
          {(final || current) && (
            <div className="card p-5 sm:p-6 space-y-4">
              <div className="flex items-center justify-between gap-3">
                <span className="label">Answer</span>
                {final ? (
                  <Badge ok={final.verified} neutral={final.refused || final.direct}>
                    {final.direct ? "No sources needed" : final.refused ? "Not in sources" : final.verified ? "Verified against sources" : "Could not fully verify"}
                  </Badge>
                ) : current?.verdict === undefined && current?.text ? <Status text="Writing…" /> : <Status text="Fact-checking…" />}
              </div>
              {final
                ? <Highlighted text={final.answer} spans={final.verified ? (audit !== "pending" && audit?.spans) || [] : best?.verdict?.spans} />
                : <Highlighted text={current!.text} />}
              {final?.direct && <p className="text-xs text-muted">This question needs no outside facts (math, logic or conversation), so it was answered directly without a source check.</p>}
              {final?.refused && <p className="text-xs text-muted">{sources.length ? "The retrieved sources don't contain this answer" : "No sources were found for this"}, so SachAI declined rather than guess.</p>}
              {final && !final.verified && !final.direct && !final.refused && <p className="text-xs text-muted">Highlighted parts could not be confirmed by the sources. Hover a highlight for its confidence.</p>}
              {final && audit && (
                <p className="text-xs text-muted flex items-center gap-1.5">
                  {audit === "pending" ? <><Loader2 size={12} className="animate-spin" /> Deep check with the trained detector…</>
                    : audit.unavailable ? "Deep check unavailable right now."
                    : audit.hallucinated ? `Deep check: the trained detector flagged ${audit.spans?.length} phrase${audit.spans?.length === 1 ? "" : "s"} (highlighted).`
                    : "Deep check: the trained detector also found no hallucination."}
                </p>
              )}
            </div>
          )}

          {/* Verification trail */}
          {attempts.some((a) => a.verdict && !a.verdict.refused) && (
            <details className="card p-5 group" open={!final || attempts.length > 1}>
              <summary className="flex items-center justify-between cursor-pointer list-none">
                <span className="label">Verification trail</span>
                <span className="text-xs text-muted">{attempts.length} attempt{attempts.length > 1 ? "s" : ""}</span>
              </summary>
              <ol className="mt-4 space-y-4">
                {attempts.map((a, i) => a.verdict && (
                  <li key={i} className="relative pl-6 border-l border-line">
                    <span className={`absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full ${a.verdict.hallucinated ? "bg-bad" : "bg-ok"}`} />
                    <div className="flex items-center gap-2 text-sm font-medium mb-1">
                      {i === 0 ? "Initial answer" : <><RotateCcw size={13} /> Retry {i}</>}
                      <span className="text-muted font-normal">· risk {Math.round(a.verdict.score * 100)}%{a.verdict.jev != null && ` · Jev support ${Math.round(a.verdict.jev * 100)}%`}</span>
                    </div>
                    <div className="text-sm text-muted"><Highlighted text={a.text} spans={a.verdict.spans} /></div>
                  </li>
                ))}
              </ol>
            </details>
          )}

          {/* Sources */}
          {sources.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 label">{mode === "web" ? <Globe size={13} /> : <FileText size={13} />} Sources · {mode === "web" ? "Wikipedia" : "Your documents"}</div>
              <div className="grid sm:grid-cols-2 gap-3">
                {sources.map((s, i) => (
                  <a key={i} href={s.url} target="_blank" rel="noreferrer" className={`card p-4 text-sm block ${s.url ? "hover:border-fg/20" : "pointer-events-none"} transition`}>
                    <div className="font-medium mb-1 truncate"><span className="text-muted mr-1.5">{i + 1}</span>{s.title}</div>
                    <p className="text-muted line-clamp-3">{s.text}</p>
                  </a>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  )
}

function Status({ text }: { text: string }) {
  return <span className="inline-flex items-center gap-2 text-xs text-muted"><Loader2 size={13} className="animate-spin" />{text}</span>
}
