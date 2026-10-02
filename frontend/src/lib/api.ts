export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000"

export interface Span { start: number; end: number; text: string; prob: number; by?: string }
export interface Verdict { hallucinated: boolean; score: number; spans: Span[]; jev: number | null; refused?: boolean }
export interface Source { title: string; text: string; url?: string }

export type AskEvent =
  | { type: "sources"; mode: "docs" | "web" | "direct"; sources: Source[] }
  | { type: "attempt"; n: number }
  | { type: "token"; n: number; text: string }
  | ({ type: "verdict"; n: number } & Verdict)
  | { type: "final"; answer: string; verified: boolean; best: number; refused?: boolean; direct?: boolean }
  | { type: "audit"; hallucinated?: boolean; spans?: Span[]; unavailable?: boolean }
  | { type: "error"; message: string }

export const sessionId = (() => {
  const k = "sachai.session"
  try {
    let id = localStorage.getItem(k)
    if (!id) localStorage.setItem(k, (id = crypto.randomUUID()))
    return id
  } catch {
    return crypto.randomUUID()
  }
})()

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const e = await res.json().catch(() => ({}))
    throw new Error(e.detail ?? `Request failed (${res.status})`)
  }
  return res.json()
}

export interface Model { id: string; label: string; provider: string }
export const listModels = () => fetch(`${API}/models`).then(json<{ default: string | null; models: Model[] }>)

export async function ask(question: string, model: string, onEvent: (e: AskEvent) => void, signal?: AbortSignal) {
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model, session_id: sessionId }),
    signal,
  })
  if (!res.ok || !res.body) return void (await json(res))
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += value
    const lines = buf.split("\n")
    buf = lines.pop()!
    for (const l of lines) if (l.trim()) onEvent(JSON.parse(l))
  }
}

export const verify = (context: string, answer: string, question = "") =>
  fetch(`${API}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, answer, question }),
  }).then(json<Verdict>)

export const upload = (files: File[]) => {
  const fd = new FormData()
  fd.append("session_id", sessionId)
  files.forEach((f) => fd.append("files", f))
  return fetch(`${API}/upload`, { method: "POST", body: fd }).then(json<{ docs: string[] }>)
}

export const listDocs = () => fetch(`${API}/files/${sessionId}`).then(json<{ docs: string[] }>)
export const clearDocs = () => fetch(`${API}/files/${sessionId}`, { method: "DELETE" }).then(json<{ docs: string[] }>)
export const warmup = () => fetch(`${API}/health`).catch(() => {})

export interface Stats {
  questions: number; verified: number; refused: number; needed_retry: number
  fixed_by_retry: number; avg_attempts: number | null; avg_latency_ms: number | null
}
export const getStats = () => fetch(`${API}/stats`).then(json<Stats>)
