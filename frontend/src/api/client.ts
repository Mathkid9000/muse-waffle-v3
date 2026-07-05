const BASE = '/api'

// Creative
export interface InfillRequest {
  xml: string
  start_measure: number
  end_measure: number
  /** 0-indexed part indices to fill; omit or null to fill all parts. */
  parts?: number[] | null
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

// Deterministic
export interface EditClefRequest {
  xml: string
  measure_num: number
  staff_num: number
  new_clef: number
}

export interface EditTimeSigRequest {
  xml: string
  measure_num: number
  /** 0=2/4, 1=3/4, 2=4/4, 3=6/8, 4=3/8, 5=12/8 */
  new_time: number
}

export interface EditKeySigRequest {
  xml: string
  measure_num: number
  new_tonic: string
  new_mode: string
}

export interface EditTempoRequest {
  xml: string
  measure_num: number
  /** New tempo in beats per minute */
  new_tempo: number
}

export interface EditInstrRequest {
  xml: string
  /** Instrument index 0-31 (see INSTR_NAMES) */
  instr: number
}

export interface EditRmStaffRequest {
  xml: string
  staff_num: number
}

export interface EditRmMeasureRequest {
  xml: string
  measure_start_num: number
  measure_end_num: number
}

// Other
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
  // Creative
  infill: (req: InfillRequest) =>
    post<{ xml: string }>('/infill', req),
  generate: (req: GenerateRequest) =>
    post<{ xml: string }>('/generate', req),
  append: (req: AppendRequest) =>
    post<{ xml: string }>('/append', req),
  streamGenerate: (req: GenerateRequest) => streamPost('/generate/stream', req),
  streamAppend: (req: AppendRequest) => streamPost('/append/stream', req),
  streamInfill: (req: InfillRequest) => streamPost('/infill/stream', req),

  // Deterministic
  editClef: (req: EditClefRequest) =>
    post<{ xml: string }>('/edit/clef', req),
  editTimeSig: (req: EditTimeSigRequest) =>
    post<{ xml: string }>('/edit/time_sig', req),
  editKeySig: (req: EditKeySigRequest) =>
    post<{ xml: string }>('/edit/key_sig', req),
  editTempo: (req: EditTempoRequest) =>
    post<{ xml: string }>('/edit/tempo', req),
  editInstr: (req: EditInstrRequest) =>
    post<{ xml: string }>('/edit/instr', req),
  editRmStaff: (req: EditRmStaffRequest) =>
    post<{ xml: string }>('/edit/rm-staff', req),
  editRmMeasure: (req: EditRmMeasureRequest) =>
    post<{ xml: string }>('/edit/rm-measure', req),
}
