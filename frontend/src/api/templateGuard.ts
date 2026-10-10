import type { AnyTemplateItem, ChordTemplateItem } from '../types';

type Chord = { root: string, quality: string }

// chord-symbol spellings -> the template's quality enum
const QUALITY_ALIASES: Record<string, string> = {
    '': 'maj', maj: 'maj', maj7: 'maj7', m: 'min', min: 'min', m7: 'min7', min7: 'min7',
    '7': 'dom7', dom7: 'dom7', o7: 'dim7', dim7: 'dim7', dim: 'dim', 'ø7': 'm7b5', m7b5: 'm7b5',
    sus: '7sus', '7sus': '7sus', sus2: 'sus2', '+': 'aug', aug: 'aug', '7+': '7aug', '7aug': '7aug',
}

// A root letter alone ("A section", "a C") is too often plain English, so a chord in free text
// must carry an accidental or a quality suffix (Abmaj7, F#m, Bb, G7).
const CHORD_RE = /\b([A-G])([b#]?)(maj7|min7|dom7|m7b5|dim7|7sus|7aug|7\+|maj|min|dim|aug|sus2|sus|m7|o7|ø7|m|7|\+)?(?![A-Za-z0-9#])/g

/** The absolute chord symbols named in free text (a prompt), in order of appearance. */
export function chordsInText(text: string): Chord[] {
    const found: Chord[] = []
    for (const [, letter, accidental, suffix] of text.matchAll(CHORD_RE)) {
        if (!accidental && suffix === undefined) continue
        const quality = QUALITY_ALIASES[suffix ?? '']
        if (!quality) continue
        found.push({ root: letter + accidental, quality })
    }
    return found
}

const sameRoot = (a: string, b: string) => a.toLowerCase() === b.toLowerCase()

/**
 * What is wrong with a freshly written template, as far as the user's chords go: a chord the
 * user named that no chord item carries, and chord items with a field omitted (which masks that
 * field, so the chord the user asked for would be left for the filler to invent).
 */
export function chordProblems(userText: string, template: AnyTemplateItem[]): string[] {
    const problems: string[] = []
    const items = template.filter((i): i is Extract<AnyTemplateItem, { kind: 'chord' }> => i.kind === 'chord')
    for (const want of chordsInText(userText)) {
        if (!items.some(i => i.root && sameRoot(i.root, want.root) && i.quality === want.quality)) {
            problems.push(`The user asked for ${want.root}${want.quality} but no chord item has root "${want.root}" and quality "${want.quality}". `
                + `Add one, with measure and pos set. Do not spell the chord out as note pitches.`)
        }
    }
    items.forEach((i, idx) => {
        const missing = (['root', 'quality', 'measure', 'pos'] as const).filter(f => i[f] == null)
        if (missing.length) problems.push(`Chord item #${idx + 1} omits ${missing.join(', ')}, which masks it so nothing constrains the filler. Set them.`)
    })
    // chords only say what harmony to write; something still has to be left for the filler to write it into
    if (!hasWritableContent(template)) {
        problems.push('The template has chords but nothing for the filler to write: no note items with pitch omitted and no masked block. '
            + 'For each instrument voice and each measure, add a note item (kind "note", voice_idx, measure, pos 0, duration = the measure length, e.g. 4 for a whole note in 4/4) and OMIT pitch.')
    }
    return problems
}

const hasWritableContent = (template: AnyTemplateItem[]) =>
    template.some(i => i.kind === 'masked' || (i.kind === 'note' && i.pitch == null))

/** Length of a measure in quarter notes, from the template's first complete time signature. */
function measureLength(template: AnyTemplateItem[]): number {
    const time = template.find((i): i is Extract<AnyTemplateItem, { kind: 'time' }> => i.kind === 'time' && !!i.numerator && !!i.denominator)
    return time ? Number(time.numerator) * 4 / Number(time.denominator) : 4
}

/**
 * Last resort once the model has been corrected and still got it wrong: add the chords the user
 * named, one per measure from the start of the first masked block (or measure 1), and give
 * incomplete chord items a place. Mutates and returns the template.
 */
export function repairChords(userText: string, template: AnyTemplateItem[]): AnyTemplateItem[] {
    const masked = template.find(i => i.kind === 'masked')
    const start = masked && masked.kind === 'masked' ? masked.measure_start : 1
    const chordItems = () => template.filter((i): i is Extract<AnyTemplateItem, { kind: 'chord' }> => i.kind === 'chord')

    for (const item of chordItems()) {
        if (item.measure == null && item.root && item.quality) item.measure = start
        if (item.pos == null && item.root && item.quality) item.pos = 0
    }
    chordsInText(userText).forEach((want, idx) => {
        if (chordItems().some(i => i.root && sameRoot(i.root, want.root) && i.quality === want.quality)) return
        template.push({ kind: 'chord', root: want.root, quality: want.quality as ChordTemplateItem['quality'], measure: start + idx, pos: 0 })
    })

    // one pitch-less note per voice under each chord, sized to the measure -- a pad of masked whole notes
    if (!hasWritableContent(template)) {
        const voices = template.flatMap(i => i.kind === 'instrument' ? [i.voice_idx] : [])
        const measures = [...new Set(chordItems().flatMap(c => c.measure != null ? [c.measure] : []))]
        const duration = measureLength(template)
        for (const measure of measures) for (const voice_idx of voices) {
            template.push({ kind: 'note', voice_idx, measure, pos: 0, duration })
        }
    }
    return template
}
