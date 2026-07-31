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
const INSTR_NAMES = [
  'Other', 'Piano', 'Harpsichord', 'Organ', 'Accordion', 'Guitar', 'Bass Guitar',
  'Violin', 'Viola', 'Cello', 'Double Bass', 'Harp', 'Flute', 'Oboe',
  'Clarinet', 'Bassoon', 'Saxophone', 'Horn', 'Trumpet', 'Trombone', 'Tuba',
  'Percussion', 'Timpani', 'Celesta', 'Soprano', 'Mezzo-Soprano', 'Alto',
  'Tenor', 'Baritone', 'Bass Voice', 'Voice', 'Choir',
]
const KEY_LABELS = [
  '6♭', '5♭', '4♭', '3♭', '2♭', '1♭', 'C', '1♯', '2♯', '3♯', '4♯', '5♯', '6♯',
]
const TIME_SIG_LABELS = ['2/4', '3/4', '4/4', '6/8', '3/8', '12/8']
const TEMPO_LABELS = [
  '≈20', '≈40', '≈52', '≈63', '≈72', '≈84', '≈96', '≈108',
  '≈116', '≈126', '≈138', '≈152', '≈168', '≈192', '≈216', '≈256',
]

export async function applyClaudeEdit(tool: string, input: Record<string, unknown>, xml: string) {
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
    }

    return xml
}