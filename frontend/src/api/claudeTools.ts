import Anthropic from '@anthropic-ai/sdk';
import { getSharedToolkit } from '../hooks/useVerovio';
import type { ToolUseBlock } from '@anthropic-ai/sdk/resources';
import type { ThoughtProcessEvent, ThoughtProcessTask } from '../components/ScoreViewer/InscorePrompt'

// Local dev only: the API key is bundled into the client, so never deploy this
// build publicly — move the call behind the FastAPI backend before sharing.
const anthropic = new Anthropic({
    apiKey: import.meta.env.VITE_ANTHROPIC_API_KEY,
    dangerouslyAllowBrowser: true,
});

export interface ScoreEdit {
    tool: string;
    input: unknown;
}

// The user's live selection on the score, mirrored from the app store. Lets Claude
// resolve requests like "switch the selected staff to alto clef" via the read_score tool.
export interface ScoreSelection {
    start?: number;               // 1-based first selected measure
    end?: number;                 // 1-based last selected measure
    parts?: number[] | null;      // 0-based staff indices; null/empty = all staves
}

// Cap on the read → edit agentic turns in stage 2, so a model that keeps asking to
// read can never spin forever (each turn is one API call).
const MAX_STAGE2_TURNS = 4;

export async function promptAnthropic(prompt: string, xml: string, pushCallback: (thought: ThoughtProcessEvent | ThoughtProcessTask) => void, pushConversationName: (title: string) => void, selection: ScoreSelection | null): Promise<any> {
    // Stage 1 — decide which edit tool TYPES the request needs. Cheap: only the tiny
    // trigger_tool schema is sent, and the reply is just a list of tool names.
    pushCallback({ task: 'Selecting tools' })
    const message = await anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1024,
        tools: TOOL_INTROS as unknown as Anthropic.Messages.ToolUnion[],
        system: "Your ONLY job is to classify which edit types the request needs, never to act on it or ask questions. "
            + "Do not worry about missing details like which measures/staves/values are meant. "
            + "References like \"these measures\", \"the selected staff\", or even no given location always refer to that selection, so still classify them. "
            + "If nothing matches an edit type, return an empty list. If the user asks for a generation, the generate_score tool encapsulates most types of score edits and may also add new measures.",
        messages: [
            { role: 'user', content: prompt }
        ],
        tool_choice: { type: "tool", name: "trigger_tool" }
    });
    console.log(message)

    const trigger = message.content.find(
        (i): i is ToolUseBlock => i.type === "tool_use" && i.name === "trigger_tool"
    );
    if (!trigger) return [];

    // Unique tool types the model selected, keeping only ones we actually implement.
    const requested = ((trigger.input as { tools?: string[] }).tools ?? [])
        .filter((t): t is keyof typeof TOOLS => t in TOOLS);
    const toolTypes = Array.from(new Set(requested));
    if (toolTypes.length === 0) return [];

    pushConversationName((trigger.input as { promptTitle?: string }).promptTitle ?? 'No name generated')
    pushCallback({ title: `Selected ${toolTypes.length} tools`, input: toolTypes, tokensIn: message.usage.input_tokens, tokensOut: message.usage.output_tokens })
    pushCallback({ task: 'Querying next steps' })

    // Build the tool schema + context once per selected type — send only what stage 2 needs.
    // read_score is always offered on top so Claude can look up any OTHER aspect on demand
    // (e.g. locate the measure where the key changes) without us pre-sending everything.
    const tools = [...toolTypes.map(t => TOOLS[t]), READ_SCORE_TOOL];
    const context = [
        ...toolTypes
            .map(t => getPromptContext(t, xml))
            .filter((c): c is string => !!c),
        READ_SCORE_HINT(selection, xml),
    ].join("\n\n");

    // Stage 2 — an agentic read → edit loop. tool_choice "auto" lets Claude either call
    // read_score (to gather context we didn't pre-send) or emit the edit tool calls.
    // We keep looping while it reads, and stop on the first turn that produces no reads.
    const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt }];
    let edits: ScoreEdit[] = [];
    let totalInputUsage = message.usage.input_tokens;
    let totalOutputUsage = message.usage.output_tokens;

    for (let turn = 0; turn < MAX_STAGE2_TURNS; turn++) {
        const commandMessage = await anthropic.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 1024,
            tools: tools as unknown as Anthropic.Messages.ToolUnion[],
            system: context,
            messages,
            tool_choice: { type: "auto" }
        });
        totalInputUsage += commandMessage.usage.input_tokens;
        totalOutputUsage += commandMessage.usage.output_tokens;

        const toolUses = commandMessage.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
        const reads = toolUses.filter(b => b.name === "read_score");
        const tokensIn = commandMessage.usage.input_tokens
        const tokensOut = commandMessage.usage.output_tokens

        // No read requested this turn → whatever tool calls came back are the edits. Done.
        if (reads.length === 0) {
            edits = toolUses.map(b => ({ tool: b.name, input: b.input }));

            pushCallback({ title: `Found ${edits.length} edits`, input: edits.map(e => ({ tool: e.tool, ...(e.input as Record<string, any>) })), tokensIn, tokensOut })
            pushCallback({ task: 'Querying next steps' })
            console.log(context, commandMessage, edits);
            break;
        }

        // Claude asked to read: answer each read_score call locally, then let it continue.
        reads.forEach(r => {
            const queries = (r.input as { queries?: ReadQuery[] }).queries
            queries?.forEach(query => {
                const result = getReadResults([ query ], xml, selection)
                pushCallback({ title: `Queried for ${query}`, input: result, tokensIn: tokensIn / queries.length, tokensOut: tokensOut / queries.length })
                pushCallback({ task: 'Querying next steps' })
            })
        })

        messages.push({ role: 'assistant', content: commandMessage.content });
        messages.push({
            role: 'user',
            content: reads.map(r => ({
                type: 'tool_result' as const,
                tool_use_id: r.id,
                content: getReadResults((r.input as { queries?: ReadQuery[] }).queries ?? [], xml, selection),
            })),
        });
    }

    // pushCallback(`Thinking complete...`, `${totalInputUsage} in / ${totalOutputUsage} out`, 0, 0)
    return [ edits, totalInputUsage, totalOutputUsage ]
}

// The aspects of the score read_score can look up. Each maps to a neutral description
// built from the same walkers used for the edit-tool context.
type ReadQuery = "clef" | "key_signature" | "time_signature" | "tempo" | "staves" | "measure_count" | "selection";

// Resolve one read_score call: describe every requested aspect and join into one result.
function getReadResults(queries: ReadQuery[], xml: string, selection?: ScoreSelection | null): string {
    const unique = Array.from(new Set(queries));
    if (unique.length === 0) return "No aspects were requested.";
    return unique.map(q => getReadResult(q, xml, selection)).join("\n\n");
}

function getReadResult(query: ReadQuery, xml: string, selection?: ScoreSelection | null): string {
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
        case "selection":
            return describeSelection(selection);
    }
}

// Join a header with its body lines, or return the empty note when nothing was found.
function readHeaded(header: string, lines: string[], empty: string): string {
    return lines.length === 0 ? empty : [`${header}:`, ...lines].join("\n");
}

// Turn the app's live selection into prose Claude can act on (e.g. "the selected staff").
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

function getPromptContext(promptType: string, xml: string): string | undefined {
    if (xml === "") return "No current score exists."
    switch (promptType) {
        case "edit_clef":
            return getContext_edit_clef(xml);
        case "edit_time_sig":
            return getContext_scoreElement(xml, "meterSig", "time signature", "edit_time_sig");
        case "edit_key_sig":
            return getContext_scoreElement(xml, "keySig", "key signature", "edit_key_sig");
        case "edit_tempo":
            return getContext_scoreElement(xml, "tempo", "tempo", "edit_tempo");
        case "edit_add_instrument":
            return getContext_staves(xml, "edit_add_instrument");
        case "edit_delete_staff":
            return getContext_staves(xml, "edit_delete_staff");
        case "edit_delete_measures":
            return getContext_measures(xml);
        case "generate_score":
            return getContext_generate(xml);
    }
}

// generate_score needs both halves at once: which staves exist (0-based staff_num) and where the
// score ends, so Claude can place the block without a read_score round trip.
function getContext_generate(xml: string): string {
    const staves = describeStaves(xml);
    if (staves.length === 0) return "The score is empty — a generation starts a new score from scratch, so name the instruments.";

    const n = countMeasures(xml);
    return [
        `The score has ${n} measure${n === 1 ? "" : "s"} (1-based). measure_start_num is the first generated measure `
            + `and generation_length is how many; ${n + 1} continues past the end.`,
        "Generate every staff unless the user asked for specific ones; their instruments are already known, so leave `instrument` out.",
        "",
        ...staves,
    ].join("\n");
}

function getContext_edit_clef(xml: string): string {
    const lines = describeScoreElements(xml, "clef");
    if (lines.length === 0) return "No staves or clefs were found in the score.";

    return [
        "Current clef layout of the score. Use staff_num as the 0-based staff_num for the edit_clef tool; measures are 1-based.",
        "Each staff lists its clef at measure 1 and every measure where the clef changes.",
        "",
        ...lines,
    ].join("\n");
}

// Whole-score element context (time signature / key signature / tempo). Lists the value
// at measure 1 and every measure it changes, so Claude can pick the right measure_num.
function getContext_scoreElement(xml: string, type: ScoreElementType, label: string, tool: string): string {
    const lines = describeScoreElements(xml, type);
    if (lines.length === 0) return `No ${label} information was found in the score.`;

    return [
        `Current ${label} of the score. Measures are 1-based; use measure_num for the ${tool} tool.`,
        `Listed at measure 1 and every measure where the ${label} changes.`,
        "",
        ...lines,
    ].join("\n");
}

// Staff roster context — used both for adding an instrument (new staff appended at the end)
// and for deleting a staff (0-based staff_num).
function getContext_staves(xml: string, tool: "edit_add_instrument" | "edit_delete_staff"): string {
    const lines = describeStaves(xml);
    if (lines.length === 0) return "No staves were found in the score.";

    const header = tool === "edit_delete_staff"
        ? "Current staves in the score. Use the 0-based staff_num for the edit_delete_staff tool."
        : "Current staves in the score. A new instrument is added as an extra staff at the end.";

    return [header, "", ...lines].join("\n");
}

// Measure-count context for edit_delete_measures.
function getContext_measures(xml: string): string {
    const n = countMeasures(xml);
    if (n === 0) return "No measures were found in the score.";

    return `The score has ${n} measure${n === 1 ? "" : "s"}, numbered 1 to ${n} (1-based). `
        + "Use measure_start_num and measure_end_num (both inclusive) for the edit_delete_measures tool.";
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

// Read-only lookup tool. Returns a compact description of the requested aspects of the
// current score so Claude can find WHERE to apply an edit without us pre-sending everything.
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

const TOOL_INTROS = [
    {
        name: "trigger_tool",
        description: "Edit the score by choosing the tools needed. Include every distinct edit type the user is asking for.",
        input_schema: {
            type: "object",
            properties: {
                tools: {
                    type: "array",
                    items: {
                        type: "string",
                        enum: [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures",
                            "generate_score"
                        ]
                    },
                    description: "Each distinct type of edit the user wants to make. List a type once even if it will be applied several times."
                },
                promptTitle: {
                    type: "string",
                    description: "Describe the topic of the request as a short conversation title in as little words as possible. Ex: \"Change to 3/4\", \"General score changes\", or \"Edit key signature\""
                }
            },
            required: [ "tools", "promptTitle" ]
        }
    },
]

const TOOLS = {
    "edit_clef": {
        name: "edit_clef",
        description: "Edit the clef of a staff in the score by providing the measure number and new clef",
        input_schema: {
            type: "object",
            properties: {
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new clef into. Begins at 1."
                },
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to insert the new clef into. Begins at 0."
                },
                new_clef: {
                    type: "string",
                    enum: [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb", "none" ],
                    description: "The new clef to insert. None to remove clef."
                }
            }
        }
    },
    "edit_time_sig": {
        name: "edit_time_sig",
        description: "Edit the time signature of the score by providing the measure number and new time",
        input_schema: {
            type: "object",
            properties: {
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new time signature into. Begins at 1."
                },
                new_time: {
                    type: "string",
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8" ],
                    description: "The new time signature to insert"
                }
            }
        }
    },
    "edit_key_sig": {
        name: "edit_key_sig",
        description: "Edit the key signature of the score by providing the measure number and new key",
        input_schema: {
            type: "object",
            properties: {
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new key signature into. Begins at 1."
                },
                tonic: {
                    type: "string",
                    description: "The tonic note of the new key. Ex: C, Bb, E, F#"
                },
                mode: {
                    type: "string",
                    description: "The mode of the new key. Ex: major, minor, mixolydian"
                }
            }
        }
    },
    "edit_tempo": {
        name: "edit_tempo",
        description: "Edit the tempo of the score by providing the measure number and new tempo",
        input_schema: {
            type: "object",
            properties: {
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new tempo into. Begins at 1."
                },
                new_tempo: {
                    type: "integer",
                    description: "The new tempo in beats per minute"
                }
            }
        }
    },
    "edit_add_instrument": {
        name: "edit_add_instrument",
        description: "Add an instrument to the score by providing the new instrument",
        input_schema: {
            type: "object",
            properties: {
                new_instrument: {
                    type: "string",
                    enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                    description: "The new instrument to add"
                }
            }
        }
    },
    "edit_delete_staff": {
        name: "edit_delete_staff",
        description: "Remove a staff on the score",
        input_schema: {
            type: "object",
            properties: {
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to delete. Begins at 0."
                }
            }
        }
    },
    "edit_delete_measures": {
        name: "edit_delete_measures",
        description: "Remove a range of measures on the score (inclusive)",
        input_schema: {
            type: "object",
            properties: {
                measure_start_num: {
                    type: "integer",
                    description: "The first measure of the range to be deleted"
                },
                measure_end_num: {
                    type: "integer",
                    description: "The last measure of the range to be deleted"
                }
            }
        }
    },
    "generate_score": {
        name: "generate_score",
        description: "Uses a transformer model to generate a new score if no current score exists. Else, inserts into the given location. Do not include key, time_signature, or tempo when these attributes should remain the same.",
        input_schema: {
            type: "object",
            properties: {
                measure_start_num: {
                    type: "integer",
                    description: "The first measure to insert the generation into"
                },
                generation_staffs: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            staff_num: {
                                type: "integer",
                                description: "The index of the staff"
                            },
                            instrument: {
                                type: "string",
                                enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                                description: "The name of the instrument to generate. Only include if the instruments aren't already identified by the staff_num (e.g. no current score exists)"
                            }
                        },
                        required: [ "staff_num" ]
                    },
                    description: "List the staffs to be generated. Provide an empty array if no staffs exist and the user hasn't given a list of staffs or instruments"
                },
                generation_length: {
                    type: "integer",
                    description: "The number of measures to generate (multiple staffs still counts as one measure)"
                },
                overwrite_current: {
                    type: "boolean",
                    description: "True if the generation should overwrite the current measures located from measure_start_num to measure_start_num + generation_length. False means it will be inserted, creating new measures at the given location."
                },
                tonic: {
                    type: [ "string", "null" ],
                    enum: [ "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", null ],
                    description: "The tonic of the key the generation is in, spelled as the mode's own tonic (A minor is 'A', not 'C'). Null to carry on in whatever key is in effect before measure_start_num."
                },
                mode: {
                    type: [ "string", "null" ],
                    enum: [ "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian", null ],
                    description: "The mode of that key. Give it alongside tonic; the backend derives the key signature from the pair. Null is read as major."
                },
                time_signature: {
                    type: [ "string", "null" ],
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8", null ],
                    description: "The time signature the generation is in. Null to carry on in whatever time signature is in effect before measure_start_num."
                },
                tempo: {
                    type: [ "integer", "null" ],
                    description: "The tempo of the generation in beats per minute. Null to carry on at whatever tempo is in effect before measure_start_num."
                }
            },
            required: [ "measure_start_num", "generation_staffs", "generation_length", "overwrite_current", "tonic", "mode", "time_signature", "tempo" ]
        }
    }
}