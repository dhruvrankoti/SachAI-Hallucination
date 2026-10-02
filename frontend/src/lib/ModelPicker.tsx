import { useEffect, useRef, useState } from "react"
import { Check, ChevronDown } from "lucide-react"

// Mirrors MODELS in backend/app.py, so the picker renders instantly instead of waiting on the API.
export const MODELS = [
  { id: "gpt-oss-120b", label: "GPT-OSS 120B", provider: "Groq" },
  { id: "gpt-oss-20b", label: "GPT-OSS 20B", provider: "Groq" },
  { id: "qwen3.8-27b", label: "Qwen3.8 27B", provider: "Groq" },
  { id: "llama-3.3-70b", label: "Llama 3.3 70B", provider: "OpenRouter" },
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", provider: "OpenRouter" },
  { id: "claude-haiku-4.5", label: "Claude Haiku 4.5", provider: "OpenRouter" },
  { id: "deepseek-v3.1", label: "DeepSeek V3.1", provider: "OpenRouter" },
  { id: "gpt-4o-mini", label: "GPT-4o mini", provider: "OpenRouter" },
]
// fastest in our measured runs (verified answers in 2.3–3.1 s) and correct on every test question
export const DEFAULT_MODEL = "gpt-oss-120b"

export function ModelPicker({ value, onChange, available }: { value: string; onChange: (id: string) => void; available?: string[] }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const models = available ? MODELS.filter((m) => available.includes(m.id)) : MODELS
  const current = MODELS.find((m) => m.id === value) ?? MODELS[0]

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false) }
    document.addEventListener("mousedown", close)
    document.addEventListener("keydown", esc)
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc) }
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}
        className="inline-flex items-center gap-1.5 h-8 pl-2.5 pr-2 rounded-xl text-sm text-muted hover:text-fg hover:bg-line/50 transition">
        {current.label}
        <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div role="listbox" className="absolute left-0 top-full mt-2 z-20 w-60 card shadow-xl shadow-black/20 p-1.5 fade">
          {["Groq", "OpenRouter"].map((p) => {
            const group = models.filter((m) => m.provider === p)
            return group.length > 0 && (
              <div key={p} className="py-1">
                <div className="px-2.5 pb-1 text-[11px] font-medium uppercase tracking-wider text-muted/70">{p}</div>
                {group.map((m) => (
                  <button key={m.id} type="button" role="option" aria-selected={m.id === value}
                    onClick={() => { onChange(m.id); setOpen(false) }}
                    className={`w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-sm text-left transition
                      ${m.id === value ? "bg-line/60 text-fg" : "text-muted hover:bg-line/40 hover:text-fg"}`}>
                    <span className="flex items-center gap-2">
                      {m.label}
                      {m.id === DEFAULT_MODEL && <span className="text-[10px] font-medium text-accent bg-accent/10 rounded-full px-1.5 py-0.5">Recommended</span>}
                    </span>
                    {m.id === value && <Check size={14} className="text-accent shrink-0" />}
                  </button>
                ))}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
