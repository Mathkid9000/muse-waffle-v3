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

export const api = {
  infill: (req: InfillRequest) =>
    post<{ xml: string }>('/infill', req),
  generate: (req: GenerateRequest) =>
    post<{ xml: string }>('/generate', req),
  append: (req: AppendRequest) =>
    post<{ xml: string }>('/append', req),
}
