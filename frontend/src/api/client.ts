const BASE = '/api'

export interface InfillRequest {
  xml: string
  start_measure: number
  end_measure: number
  temperature?: number
  top_k?: number
  model_name?: string
  device?: string
}

export interface GenerateRequest {
  instruments: number[]
  key: number
  time_sig: number
  tempo: number
  num_bars: number
  max_new?: number
  temperature?: number
  top_k?: number
  model_name?: string
  device?: string
}

export interface AppendRequest {
  xml: string
  num_bars: number
  max_new?: number
  temperature?: number
  top_k?: number
  model_name?: string
  device?: string
}

export interface StreamEvent {
  xml?: string
  bar?: number
  total?: number
  done?: boolean
  error?: string
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

async function* streamPost(path: string, body: unknown): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const chunks = buf.split('\n\n')
    buf = chunks.pop()!
    for (const chunk of chunks) {
      const line = chunk.replace(/^data: /, '').trim()
      if (!line) continue
      const evt: StreamEvent = JSON.parse(line)
      if (evt.error) throw new Error(evt.error)
      yield evt
    }
  }
}

export const api = {
  infill: (req: InfillRequest) =>
    post<{ xml: string }>('/infill', req),
  generate: (req: GenerateRequest) =>
    post<{ xml: string }>('/generate', req),
  append: (req: AppendRequest) =>
    post<{ xml: string }>('/append', req),
  streamGenerate: (req: GenerateRequest) => streamPost('/generate/stream', req),
  streamAppend:   (req: AppendRequest)   => streamPost('/append/stream', req),
  streamInfill:   (req: InfillRequest)   => streamPost('/infill/stream', req),
}
