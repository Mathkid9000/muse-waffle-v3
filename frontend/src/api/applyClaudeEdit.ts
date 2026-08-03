import { api } from "./client"
import type { EditClefRequest, EditInstrRequest, EditKeySigRequest, EditRmMeasureRequest, EditRmStaffRequest, EditTempoRequest, EditTimeSigRequest } from "./client"

const CLEF_ENUM = ['treble', 'bass', 'alto', 'tenor', 'treble8vb', 'treble8va', 'bass8vb']
const INSTR_ENUM = [
  'other', 'piano', 'harpsichord', 'organ', 'accordion', 'guitar', 'bass_guitar',
  'violin', 'viola', 'cello', 'double_bass', 'harp', 'flute', 'oboe', 'clarinet',
  'bassoon', 'saxophone', 'horn', 'trumpet', 'trombone', 'tuba', 'percussion',
  'timpani', 'celesta', 'voice_soprano', 'voice_mezzo', 'voice_alto', 'voice_tenor',
  'voice_baritone', 'voice_bass', 'voice', 'choir',
]
const TIME_SIG_LABELS = ['2/4', '3/4', '4/4', '6/8', '3/8', '12/8']
// Tonics the model can hold, ordered as the backend's key index: 6 flats -> 6 sharps.
const KEY_ENUM = ['Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#']
// The model stores tempo in ~16 buckets of 16 bpm (see tokenizer.py's TEMPO_ tokens).
const TEMPO_BUCKET = (bpm: number) => Math.max(0, Math.min(15, Math.round(bpm / 16)))
// A label's backend index, or null for the nulls the tool allows and for anything off-enum —
// the backend reads null as "carry on with whatever is already in effect".
const index = (labels: string[], value: unknown) => {
  const i = value == null ? -1 : labels.indexOf(String(value))
  return i < 0 ? null : i
}
const clamp = (value: unknown, min: number, max: number, fallback: number) => {
  const n = Math.round(Number(value))
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback
}

/** `onStream` receives each intermediate score while a generation streams in, bar by bar. */
export async function applyClaudeEdit(tool: string, input: Record<string, unknown>, xml: string, onStream?: (xml: string) => void) {
    const args = input as Record<string, unknown>
    switch (tool) {
        case "edit_clef": {
            const new_clef = CLEF_ENUM.indexOf(String(args.new_clef))
            if (new_clef < 0) break // "none"/unknown — backend has no clef-removal case
            const req: EditClefRequest = {
                xml,
                measure_num: Number(args.measure_num),
                staff_num: Number(args.staff_num),
                new_clef,
            }
            xml = (await api.editClef(req)).xml
            break
        }
        case "edit_time_sig": {
            const new_time = TIME_SIG_LABELS.indexOf(String(args.new_time))
            if (new_time < 0) break
            const req: EditTimeSigRequest = {
                xml,
                measure_num: Number(args.measure_num),
                new_time,
            }
            xml = (await api.editTimeSig(req)).xml
            break
        }
        case "edit_key_sig": {
            const req: EditKeySigRequest = {
                xml,
                measure_num: Number(args.measure_num),
                new_tonic: String(args.tonic),
                new_mode: String(args.mode),
            }
            xml = (await api.editKeySig(req)).xml
            break
        }
        case "edit_tempo": {
            const req: EditTempoRequest = {
                xml,
                measure_num: Number(args.measure_num),
                new_tempo: Number(args.new_tempo),
            }
            xml = (await api.editTempo(req)).xml
            break
        }
        case "edit_add_instrument": {
            const instr = INSTR_ENUM.indexOf(String(args.new_instrument))
            if (instr < 0) break
            const req: EditInstrRequest = { xml, instr }
            xml = (await api.editInstr(req)).xml
            break
        }
        case "edit_delete_staff": {
            const req: EditRmStaffRequest = {
                xml,
                staff_num: Number(args.staff_num),
            }
            xml = (await api.editRmStaff(req)).xml
            break
        }
        case "edit_delete_measures": {
            const req: EditRmMeasureRequest = {
                xml,
                measure_start_num: Number(args.measure_start_num),
                measure_end_num: Number(args.measure_end_num),
            }
            xml = (await api.editRmMeasure(req)).xml
            break
        }
        case "generate_score": {
            // generation_staffs does double duty: on an existing score it picks which staves to
            // generate, and on an empty one it names the instruments the new score is built from.
            const staffs = (args.generation_staffs ?? []) as { staff_num: number, instrument?: string }[]
            // clamped to what the backend accepts: an out-of-range measure or length is ordinary
            // model sloppiness, and it should not come back as a rejected request
            for await (const evt of api.streamGenerate({
                xml,
                start_measure: clamp(args.measure_start_num, 1, 1e6, 1),
                instruments: staffs.map(s => INSTR_ENUM.indexOf(String(s.instrument))).filter(i => i >= 0),
                parts: staffs.length > 0 ? staffs.map(s => Number(s.staff_num)).filter(n => n >= 0) : null,
                key: index(KEY_ENUM, args.key),
                time_sig: index(TIME_SIG_LABELS, args.time_signature),
                tempo: args.tempo == null ? null : TEMPO_BUCKET(Number(args.tempo)),
                num_bars: clamp(args.generation_length, 1, 128, 4),
                overwrite: Boolean(args.overwrite_current),
                temperature: 0.9,
                top_k: 30,
                device: 'cpu',
            })) {
                if (evt.xml) {
                    xml = evt.xml
                    onStream?.(xml)
                }
                if (evt.done) break
            }
            break
        }
    }

    return xml
}
