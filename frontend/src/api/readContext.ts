import { getSharedToolkit } from "../hooks/useVerovio";
import { useStore } from "../store";
import type { Tab } from "../types";
import type { ScoreSelection } from "./claudeTools";

export type ReadQuery = "clef" | "key_signature" | "time_signature" | "tempo" | "staves" | "measure_count" | "selection" | "tab";

// Resolve one read_score call: describe every requested aspect and join into one result.
export function getReadResults(queries: ReadQuery[], xml: string, tab_name: string, selection?: ScoreSelection | null): string {
    const unique = Array.from(new Set(queries));
    if (unique.length === 0) return "No aspects were requested.";
    return unique.map(q => getReadResult(q, xml, tab_name, selection)).join("\n\n");
}

function getReadResult(query: ReadQuery, xml: string, tab_name: string, selection?: ScoreSelection | null): string {
    if (xml === '') return "No score exists currently. You must generate to begin."
    switch (query) {
        case "clef":
            return readHeaded("Clef layout (staff_num is 0-based, measures 1-based; listed at m1 and every change)",
                describeScoreElements(xml, "clef"), "No clefs found.");
        case "key_signature":
            return readHeaded("Key signature by measure (1-based; listed at m1 and every change)",
                describeScoreElements(xml, "keySig"), "No key signature found.");
        case "time_signature":
            return readHeaded("Time signature by measure (1-based; listed at m1 and every change)",
                describeScoreElements(xml, "meterSig"), "No time signature found.");
        case "tempo":
            return readHeaded("Tempo by measure (1-based; listed at m1 and every change)",
                describeScoreElements(xml, "tempo"), "No tempo found.");
        case "staves":
            return readHeaded("Staves (staff_num is 0-based)", describeStaves(xml), "No staves found.");
        case "measure_count": {
            const n = countMeasures(xml);
            return n === 0 ? "No measures found." : `The score has ${n} measure${n === 1 ? "" : "s"}, numbered 1 to ${n} (1-based).`;
        }
        case "selection": {
            return describeSelection(selection);
        }
        case "tab":
            return "Selected tab: " + tab_name;
    }
}

// Usage guidance for each prompt type, appended once after all tabs' context rather than
// repeated per tab. Each entry is one or more lines, in generate_score's original style.
const PROMPT_HINTS: Partial<Record<string, string[]>> = {
    edit_clef: ["Use staff_num as the 0-based staff_num for the edit_clef tool; measures are 1-based. Each staff lists its clef at measure 1 and every measure where the clef changes."],
    edit_time_sig: ["Measures are 1-based; use measure_num for the edit_time_sig tool. Listed at measure 1 and every measure where the time signature changes."],
    edit_key_sig: ["Measures are 1-based; use measure_num for the edit_key_sig tool. Listed at measure 1 and every measure where the key signature changes."],
    edit_tempo: ["Measures are 1-based; use measure_num for the edit_tempo tool. Listed at measure 1 and every measure where the tempo changes."],
    edit_add_instrument: ["A new instrument is added as an extra staff at the end."],
    edit_delete_staff: ["Use the 0-based staff_num for the edit_delete_staff tool."],
    edit_delete_measures: ["Use measure_start_num and measure_end_num (both inclusive) for the edit_delete_measures tool."],
    generate_score: [
        `measure_start_num is the first generated measure `
            + `and generation_length is how many`,
        "Generate every staff unless the user asked for specific ones; If the instruments are already known, leave `instrument` out.",
    ],
}

export function getPromptContext(promptType: string, tabs: Tab[]): string | undefined {
    const scores = tabs.filter(t => t.content && !t.template)
    const context = scores.map(tab => {
        const xml = tab.content
        if (xml === "" || !xml) return "No current score exists."
        switch (promptType) {
            case "edit_clef":
                return "Context for " + tab.name + ":\n" + getContext_edit_clef(xml);
            case "edit_time_sig":
                return "Context for " + tab.name + ":\n" + getContext_scoreElement(xml, "meterSig", "time signature");
            case "edit_key_sig":
                return "Context for " + tab.name + ":\n" + getContext_scoreElement(xml, "keySig", "key signature");
            case "edit_tempo":
                return "Context for " + tab.name + ":\n" + getContext_scoreElement(xml, "tempo", "tempo");
            case "edit_add_instrument":
                return "Context for " + tab.name + ":\n" + getContext_staves(xml);
            case "edit_delete_staff":
                return "Context for " + tab.name + ":\n" + getContext_staves(xml);
            case "edit_delete_measures":
                return "Context for " + tab.name + ":\n" + getContext_measures(xml);
            case "generate_score":
                return "Context for " + tab.name + ":\n" + getContext_generate(xml);
        }
        return ''
    }).filter(c => c !== '')
    context.push(...(PROMPT_HINTS[promptType] ?? []))
    return context.join('\n\n')
}

function readHeaded(header: string, lines: string[], empty: string): string {
    return lines.length === 0 ? empty : [`${header}:`, ...lines].join("\n");
}

function describeSelection(sel?: ScoreSelection | null): string {
    if (!sel) return "The user has not selected anything on the score.";
    const parts: string[] = [];
    if (sel.start != null && sel.end != null) {
        parts.push(sel.start === sel.end ? `measure ${sel.start}` : `measures ${sel.start} to ${sel.end} (1-based, inclusive)`);
    }
    parts.push(sel.parts && sel.parts.length > 0
        ? `staff_num ${sel.parts.join(", ")} (0-based)`
        : "all staves");
    return `The user's current selection on the score: ${parts.join(", ")}.`;
}

function getContext_generate(xml: string): string {
    const staves = describeStaves(xml);
    if (staves.length === 0) return "The score is empty — a generation starts a new score from scratch, so name the instruments.";

    const n = countMeasures(xml);
    return [
        `The score has ${n} measure${n === 1 ? "" : "s"} (1-based).`,
        ...staves,
    ].join("\n");
}

function getContext_edit_clef(xml: string): string {
    const lines = describeScoreElements(xml, "clef");
    return lines.length === 0 ? "No staves or clefs were found in the score." : lines.join("\n");
}

// Whole-score element context (time signature / key signature / tempo). Lists the value
// at measure 1 and every measure it changes, so Claude can pick the right measure_num.
function getContext_scoreElement(xml: string, type: ScoreElementType, label: string): string {
    const lines = describeScoreElements(xml, type);
    return lines.length === 0 ? `No ${label} information was found in the score.` : lines.join("\n");
}

// Staff roster context — used both for adding an instrument (new staff appended at the end)
// and for deleting a staff (0-based staff_num).
function getContext_staves(xml: string): string {
    const lines = describeStaves(xml);
    return lines.length === 0 ? "No staves were found in the score." : lines.join("\n");
}

// Measure-count context for edit_delete_measures.
function getContext_measures(xml: string): string {
    const n = countMeasures(xml);
    if (n === 0) return "No measures were found in the score.";

    return `The score has ${n} measure${n === 1 ? "" : "s"}, numbered 1 to ${n} (1-based).`;
}

// The score elements this module can search for. Each maps to an MEI element and,
// where applicable, an equivalent attribute-form on <scoreDef>/<staffDef>.
type ScoreElementType = "clef" | "keySig" | "meterSig" | "tempo";

// One track of a single element type: either a staff (clef, and staff-local key/meter)
// or the whole score (key signature, time signature, tempo). `changes` holds the value
// at measure 1 and every measure where it changes, in the element's own vocabulary.
class StaffContext {
    readonly staffNum: number;  // 0-based staff index (matches edit_clef's staff_num); -1 = whole score
    readonly instrument: string;
    readonly changes: { measure: number; value: string }[] = [];
    private lastValue: string | null = null;

    constructor(staffNum: number, instrument: string) {
        this.staffNum = staffNum;
        this.instrument = instrument;
    }

    get isScoreWide(): boolean {
        return this.staffNum < 0;
    }

    // Record `value` taking effect at `measure`, collapsing consecutive no-op repeats
    // (e.g. the same key signature echoed on every staff of a <scoreDef>).
    record(measure: number, value: string): void {
        if (value === this.lastValue) return;
        this.lastValue = value;
        this.changes.push({ measure: Math.max(measure, 1), value });
    }

    // "m1=treble, m5=bass" — measure 1 value, then each change, ordered by measure.
    describeChanges(): string {
        return this.changes
            .slice()
            .sort((a, b) => a.measure - b.measure)
            .map(c => `m${c.measure}=${c.value}`)
            .join(", ");
    }
}

// Read a direct child <label>'s text, if present.
function childLabel(el: Element): string {
    for (const c of Array.from(el.children)) {
        if (c.localName === "label") return (c.textContent ?? "").trim();
    }
    return "";
}

// Resolve the instrument name for an MEI <staffDef>: its own label, else the
// nearest enclosing <staffGrp> label (how multi-staff instruments like piano are grouped).
function staffDefLabel(staffDef: Element): string {
    const own = staffDef.getAttribute("label") || childLabel(staffDef);
    if (own) return own;
    let p = staffDef.parentElement;
    while (p) {
        if (p.localName === "staffGrp") {
            const l = p.getAttribute("label") || childLabel(p);
            if (l) return l;
        }
        p = p.parentElement;
    }
    return `Staff ${staffDef.getAttribute("n") ?? "?"}`;
}

// Map MEI clef attributes (shape / line / octave displacement) to this app's clef names.
function meiClefName(shape: string, line: string, dis: string, disPlace: string): string {
    shape = shape.toUpperCase();
    const down = dis === "8" && disPlace === "below";
    const up = dis === "8" && disPlace === "above";
    switch (shape) {
        case "G": return down ? "treble8vb" : up ? "treble8va" : "treble";
        case "F": return down ? "bass8vb" : "bass";
        case "C": return line === "3" ? "alto" : line === "4" ? "tenor" : `C-clef (line ${line})`;
        case "PERC":
        case "PERCUSSION": return "percussion";
        case "TAB": return "tab";
        default: return shape ? `${shape}${line}` : "none";
    }
}

// MEI key signature `@sig` ("0", "2s", "3f", …) -> "no sharps or flats" / "N sharp(s)" / "N flat(s)".
function meiKeySigName(sig: string): string {
    sig = sig.trim();
    if (!sig || sig === "0") return "no sharps or flats";
    const m = /^(\d+)([sf])$/.exec(sig);
    if (!m) return sig; // e.g. "mixed" or an already-named value — pass through unchanged
    const count = parseInt(m[1], 10);
    const kind = m[2] === "s" ? "sharp" : "flat";
    return `${count} ${kind}${count === 1 ? "" : "s"}`;
}

// MEI meter (count / unit / sym) -> "4/4", "common (4/4)", "cut (2/2)". Empty when unspecified.
function meiMeterName(count: string, unit: string, sym: string): string {
    if (sym === "common") return "common (4/4)";
    if (sym === "cut") return "cut (2/2)";
    if (count && unit) return `${count}/${unit}`;
    return "";
}

// MEI <tempo>: prefer its text ("Allegro"), append the metronome bpm when present.
function meiTempoName(el: Element): string {
    const bpmAttr = el.getAttribute("midi.bpm") || el.getAttribute("mm");
    const bpm = bpmAttr ? Math.round(parseFloat(bpmAttr)) : NaN;
    const text = (el.textContent ?? "").replace(/\s+/g, " ").trim();
    if (text && !Number.isNaN(bpm)) return `${text} (${bpm} bpm)`;
    if (text) return text;
    if (!Number.isNaN(bpm)) return `${bpm} bpm`;
    return "";
}

// Per-element-type recipe for the generic walker: which MEI element carries the value,
// how to read it from a standalone element vs. the attribute-form on <scoreDef>/<staffDef>,
// whether it belongs to one staff or the whole score, and whether courtesy shifting applies.
interface ElementDescriptor {
    tag: string;                 // MEI localName of the standalone element (<clef>, <keySig>, …)
    scope: "staff" | "score";    // clef is per-staff; key/meter/tempo apply to the whole score
    label: string;               // human name for prose ("clef", "key signature", …)
    courtesy: boolean;           // apply the end-of-measure courtesy shift (clef only)
    fromElement(el: Element): string | null;    // read the standalone element, null if not applicable
    fromDefAttrs(def: Element): string | null;  // read the attr-form off a def, null if absent
}

const DESCRIPTORS: Record<ScoreElementType, ElementDescriptor> = {
    clef: {
        tag: "clef",
        scope: "staff",
        label: "clef",
        courtesy: true,
        fromElement: el => {
            const shape = el.getAttribute("shape");
            return shape
                ? meiClefName(shape, el.getAttribute("line") ?? "", el.getAttribute("dis") ?? "", el.getAttribute("dis.place") ?? "")
                : null;
        },
        fromDefAttrs: def => {
            const shape = def.getAttribute("clef.shape");
            return shape
                ? meiClefName(shape, def.getAttribute("clef.line") ?? "", def.getAttribute("clef.dis") ?? "", def.getAttribute("clef.dis.place") ?? "")
                : null;
        },
    },
    keySig: {
        tag: "keySig",
        scope: "score",
        label: "key signature",
        courtesy: false,
        fromElement: el => {
            const sig = el.getAttribute("sig");
            return sig != null ? meiKeySigName(sig) : null;
        },
        fromDefAttrs: def => {
            const sig = def.getAttribute("key.sig") ?? def.getAttribute("keysig");
            return sig != null ? meiKeySigName(sig) : null;
        },
    },
    meterSig: {
        tag: "meterSig",
        scope: "score",
        label: "time signature",
        courtesy: false,
        fromElement: el =>
            meiMeterName(el.getAttribute("count") ?? "", el.getAttribute("unit") ?? "", el.getAttribute("sym") ?? "") || null,
        fromDefAttrs: def => {
            const count = def.getAttribute("meter.count");
            const unit = def.getAttribute("meter.unit");
            const sym = def.getAttribute("meter.sym");
            if (count == null && unit == null && sym == null) return null;
            return meiMeterName(count ?? "", unit ?? "", sym ?? "") || null;
        },
    },
    tempo: {
        tag: "tempo",
        scope: "score",
        label: "tempo",
        courtesy: false,
        fromElement: el => meiTempoName(el) || null,
        fromDefAttrs: () => null, // tempo is a control event, never an attribute-form on a def
    },
};

// Normalize whatever `currentXml` holds — raw MusicXML, or a '__MXL__'+base64 ZIP —
// into MEI using the already-loaded Verovio toolkit (Verovio decodes both forms).
function toMei(data: string): string {
    const tk = getSharedToolkit() as unknown as {
        loadData: (d: string) => boolean;
        loadZipDataBase64: (d: string) => boolean;
        getMEI: (opts?: object) => string;
    } | null;
    if (!tk) throw new Error("Verovio is not ready yet — open a score first.");

    const ok = data.startsWith("__MXL__")
        ? tk.loadZipDataBase64(data.slice(7))
        : tk.loadData(data);
    if (!ok) throw new Error("Verovio could not load the score.");

    return tk.getMEI();
}

// Note-bearing MEI elements — used to tell a genuine mid-measure clef from an
// end-of-measure "courtesy" clef (which Verovio parks at the end of the prior measure).
const EVENT_TAGS = new Set(["note", "rest", "chord", "mRest", "multiRest", "space", "mSpace", "beam", "tuplet", "fTrem", "bTrem", "halfmRpt"]);

// Does any note/rest/chord follow this clef within its layer? If not, the clef is
// a courtesy at the measure's end and actually takes effect in the next measure.
function eventFollows(clefEl: Element): boolean {
    for (let s = clefEl.nextElementSibling; s; s = s.nextElementSibling) {
        if (EVENT_TAGS.has(s.localName)) return true;
        if (s.getElementsByTagNameNS("*", "note")[0] || s.getElementsByTagNameNS("*", "rest")[0]) return true;
    }
    return false;
}

// Key used for the single whole-score track (key signature, time signature, tempo),
// separate from real 1-based MEI staff numbers.
const SCORE_KEY = 0;

// Walk the MEI document once and extract, per track, the value of `type` at measure 1
// and at every measure where it changes. Staff-scoped types (clef) yield one track per
// staff; score-scoped types (key/meter/tempo) yield a single whole-score track.
function extractStaffElements(mei: string, type: ScoreElementType): StaffContext[] {
    const desc = DESCRIPTORS[type];
    const doc = new DOMParser().parseFromString(mei, "application/xml");

    // Instrument name per staff number, from the initial <scoreDef>.
    const instrument = new Map<number, string>();
    const firstScoreDef = doc.getElementsByTagNameNS("*", "scoreDef")[0];
    if (firstScoreDef) {
        for (const sd of Array.from(firstScoreDef.getElementsByTagNameNS("*", "staffDef"))) {
            const n = parseInt(sd.getAttribute("n") ?? "", 10);
            if (!Number.isNaN(n)) instrument.set(n, staffDefLabel(sd));
        }
    }

    // Tracks keyed by staff number (>= 1) or SCORE_KEY for the whole-score track.
    const tracks = new Map<number, StaffContext>();
    let currentMeasure = 0; // 0 = before measure 1 (initial <scoreDef>/<staffDef> values)

    const trackFor = (key: number): StaffContext => {
        let ctx = tracks.get(key);
        if (!ctx) {
            ctx = key === SCORE_KEY
                ? new StaffContext(-1, "whole score")
                : new StaffContext(key - 1, instrument.get(key) ?? `Staff ${key}`);
            tracks.set(key, ctx);
        }
        return ctx;
    };

    // A def (<scoreDef>/<staffDef>) or courtesy clef sitting after measure k takes
    // effect at measure k+1; before any measure it is the measure-1 initial value.
    const defMeasure = () => (currentMeasure === 0 ? 1 : currentMeasure + 1);

    const walk = (el: Element, staffCtx: number | null) => {
        let staff = staffCtx;
        switch (el.localName) {
            case "measure": {
                currentMeasure += 1;
                break;
            }
            case "staff": {
                const n = parseInt(el.getAttribute("n") ?? "1", 10);
                if (!Number.isNaN(n)) staff = n;
                break;
            }
            case "scoreDef": {
                // Whole-score attribute-form, e.g. key.sig / meter.count on the <scoreDef> itself.
                if (desc.scope === "score") {
                    const v = desc.fromDefAttrs(el);
                    if (v != null) trackFor(SCORE_KEY).record(defMeasure(), v);
                }
                break;
            }
            case "staffDef": {
                const n = parseInt(el.getAttribute("n") ?? "", 10);
                if (!Number.isNaN(n)) staff = n;
                // Attribute-form on the staffDef (clef.shape, key.sig, meter.count, …). Verovio
                // commonly emits key/meter here per staff; route those to the whole-score track
                // (identical values across staves collapse via record()'s dedup).
                const v = desc.fromDefAttrs(el);
                if (v != null) {
                    if (desc.scope === "score") trackFor(SCORE_KEY).record(defMeasure(), v);
                    else if (!Number.isNaN(n)) trackFor(n).record(defMeasure(), v);
                }
                break;
            }
            default: {
                if (el.localName === desc.tag) {
                    const v = desc.fromElement(el);
                    if (v != null) {
                        const parent = el.parentElement?.localName;
                        if (desc.scope === "score") {
                            // Child of a mid-score <scoreDef> => next measure; otherwise a control
                            // event (e.g. <tempo>) anchored in the current measure.
                            const measure = parent === "scoreDef" ? defMeasure() : (currentMeasure || 1);
                            trackFor(SCORE_KEY).record(measure, v);
                        } else if (staff != null) {
                            // Inside a <staffDef> => initial value. Inside a <layer>, a courtesy
                            // (no note/rest after it) actually takes effect in the next measure.
                            let measure: number;
                            if (parent === "staffDef") {
                                measure = defMeasure();
                            } else {
                                measure = desc.courtesy && !eventFollows(el) ? currentMeasure + 1 : currentMeasure;
                            }
                            trackFor(staff).record(measure, v);
                        }
                    }
                }
            }
        }
        for (const child of Array.from(el.children)) walk(child, staff);
    };

    if (doc.documentElement) walk(doc.documentElement, null);

    return Array.from(tracks.values()).sort((a, b) => a.staffNum - b.staffNum);
}

// Render the extracted tracks of one element type into one line each — the body of the
// context string handed to Claude. Empty array means nothing of that type was found.
function describeScoreElements(xml: string, type: ScoreElementType): string[] {
    const tracks = extractStaffElements(toMei(xml), type);
    return tracks.map(t => {
        const who = t.isScoreWide ? "whole score" : `staff_num ${t.staffNum} — ${t.instrument}`;
        return `${who}: ${t.describeChanges()}`;
    });
}

// One line per staff ("staff_num 0 — Piano"), ordered by 0-based staff_num. Reads the
// initial <scoreDef> so the roster reflects every declared staff, empty or not.
function describeStaves(xml: string): string[] {
    const doc = new DOMParser().parseFromString(toMei(xml), "application/xml");
    const scoreDef = doc.getElementsByTagNameNS("*", "scoreDef")[0];
    if (!scoreDef) return [];

    return Array.from(scoreDef.getElementsByTagNameNS("*", "staffDef"))
        .map(sd => ({ n: parseInt(sd.getAttribute("n") ?? "", 10), sd }))
        .filter(({ n }) => !Number.isNaN(n))
        .sort((a, b) => a.n - b.n)
        .map(({ n, sd }) => `staff_num ${n - 1} — ${staffDefLabel(sd)}`);
}

// Highest 1-based measure number in the score (0 when there are no measures).
function countMeasures(xml: string): number {
    const doc = new DOMParser().parseFromString(toMei(xml), "application/xml");
    return doc.getElementsByTagNameNS("*", "measure").length;
}

// Extra line appended to the stage-2 system prompt telling Claude the read tool exists.
const READ_SCORE_HINT = (selection: ScoreSelection | null, xml: string) => [
    "Call read_score FIRST whenever needed information was not given —",
    "Ex: locating the measure where something happens (\"where the key changes to D major\").",
    "If the user's request requires on-screen selection (\"the selected staff\", \"here\"), or if no location if given, use the selection. ",
    "If all staves are seleced, apply to all staves. If a range is selected and only one measure is allowed, use to start of the range per each staff selected. ", 
    getReadResults([ 'selection' ], xml, selection),
    "Request only the aspects you need.",
    "Once you have enough information, emit the edit tool call(s) and stop calling read_score.",
].join(" ");

const READ_SCORE_TOOL = {
    name: "read_score",
    description: "Look up specific aspects of the current score before editing. Use it to find where something occurs (e.g. the measure a key/time/tempo changes) or the measure and staff the user has selected. Request only what you need.",
    input_schema: {
        type: "object",
        properties: {
            queries: {
                type: "array",
                items: {
                    type: "string",
                    enum: [ "clef", "key_signature", "time_signature", "tempo", "staves", "measure_count", "selection" ]
                },
                description: "Aspects to read. clef=per-staff clef changes; key_signature/time_signature/tempo=value at measure 1 and every measure it changes; staves=instrument roster with 0-based staff_num; measure_count=number of measures; selection=the measure(s) and staff/staves the user has selected on the score."
            }
        },
        required: [ "queries" ]
    }
}