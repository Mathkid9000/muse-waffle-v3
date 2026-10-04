import { useState, useRef } from 'react'
import { useStore } from '../store'
import { getCurrentMidi, getMeasureStartTimes } from './useVerovio'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any

let toneCache: typeof import('tone') | null = null
let midiCache: typeof import('@tonejs/midi') | null = null

async function libs() {
  if (!toneCache || !midiCache) {
    ;[toneCache, midiCache] = await Promise.all([import('tone'), import('@tonejs/midi')])
  }
  return { Tone: toneCache!, Midi: midiCache!.Midi }
}

// GM program number → FluidR3_GM soundfont folder name.
// Programs 120–127 are non-melodic sound effects with no reliable samples; fall back to piano.
const GM_SOUNDFONTS: string[] = [
  /* 0-7   Piano          */ 'acoustic_grand_piano','bright_acoustic_piano','electric_grand_piano','honkytonk_piano','electric_piano_1','electric_piano_2','harpsichord','clavinet',
  /* 8-15  Chromatic Perc */ 'celesta','glockenspiel','music_box','vibraphone','marimba','xylophone','tubular_bells','dulcimer',
  /* 16-23 Organ          */ 'drawbar_organ','percussive_organ','rock_organ','church_organ','reed_organ','accordion','harmonica','tango_accordion',
  /* 24-31 Guitar         */ 'acoustic_guitar_nylon','acoustic_guitar_steel','electric_guitar_jazz','electric_guitar_clean','electric_guitar_muted','overdriven_guitar','distortion_guitar','guitar_harmonics',
  /* 32-39 Bass           */ 'acoustic_bass','electric_bass_finger','electric_bass_pick','fretless_bass','slap_bass_1','slap_bass_2','synth_bass_1','synth_bass_2',
  /* 40-47 Strings        */ 'violin','viola','cello','contrabass','tremolo_strings','pizzicato_strings','orchestral_harp','timpani',
  /* 48-55 Ensemble       */ 'string_ensemble_1','string_ensemble_2','synth_strings_1','synth_strings_2','choir_aahs','voice_oohs','synth_voice','orchestra_hit',
  /* 56-63 Brass          */ 'trumpet','trombone','tuba','muted_trumpet','french_horn','brass_section','synth_brass_1','synth_brass_2',
  /* 64-71 Reed           */ 'soprano_sax','alto_sax','tenor_sax','baritone_sax','oboe','english_horn','bassoon','clarinet',
  /* 72-79 Pipe           */ 'piccolo','flute','recorder','pan_flute','blown_bottle','shakuhachi','whistle','ocarina',
  /* 80-87 Synth Lead     */ 'lead_1_square','lead_2_sawtooth','lead_3_calliope','lead_4_chiff','lead_5_charang','lead_6_voice','lead_7_fifths','lead_8_bass_lead',
  /* 88-95 Synth Pad      */ 'pad_1_new_age','pad_2_warm','pad_3_polysynth','pad_4_choir','pad_5_bowed','pad_6_metallic','pad_7_halo','pad_8_sweep',
  /* 96-103 Synth FX      */ 'fx_1_rain','fx_2_soundtrack','fx_3_crystal','fx_4_atmosphere','fx_5_brightness','fx_6_goblins','fx_7_echoes','fx_8_sci-fi',
  /* 104-111 Ethnic       */ 'sitar','banjo','shamisen','koto','kalimba','bagpipe','fiddle','shanai',
  /* 112-119 Percussive   */ 'tinkle_bell','agogo','steel_drums','woodblock','taiko_drum','melodic_tom','synth_drum','reverse_cymbal',
  /* 120-127 Sound FX     */ 'acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano','acoustic_grand_piano',
]

// Anchor notes loaded per instrument: every 3 semitones across C2–C7 (21 notes).
// Tone.js Sampler pitch-shifts from the nearest anchor, so 3-semitone spacing is plenty.
const SAMPLE_NOTES = [
  'C2','Eb2','Gb2','A2',
  'C3','Eb3','Gb3','A3',
  'C4','Eb4','Gb4','A4',
  'C5','Eb5','Gb5','A5',
  'C6','Eb6','Gb6','A6',
  'C7',
]

// Survives across play/stop cycles — never disposed.
const samplerCache = new Map<string, AnyNode>()

function gmName(instrNum: number): string {
  return GM_SOUNDFONTS[instrNum] ?? 'acoustic_grand_piano'
}

function loadSampler(name: string, Tone: typeof import('tone')): AnyNode {
  if (samplerCache.has(name)) return samplerCache.get(name)!

  const urls: Record<string, string> = {}
  for (const note of SAMPLE_NOTES) {
    // Tone.js uses 'F#4', CDN filenames use 'Fs4'
    const filename = note.replace('#', 's')
    urls[note] = `${filename}.mp3`
  }

  const sampler = new Tone.Sampler({
    urls,
    baseUrl: `https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/${name}-mp3/`,
  })

  samplerCache.set(name, sampler)
  return sampler
}

function toDb(gain: number) {
  return 20 * Math.log10(Math.max(0.001, gain))
}

export function usePlayback() {
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const [volume, setVolumeState] = useState(0.8)
  const synthsRef = useRef<AnyNode[]>([])
  const reverbRef = useRef<AnyNode>(null)

  function setVolume(v: number) {
    setVolumeState(v)
    if (toneCache) toneCache.getDestination().volume.value = toDb(v)
  }

  function cleanup(Tone: typeof import('tone')) {
    Tone.getTransport().stop()
    Tone.getTransport().cancel()
    // Disconnect (but do NOT dispose) cached samplers
    const unique = new Set(synthsRef.current)
    for (const s of unique) {
      try { s.releaseAll(); s.disconnect() } catch { /* already disconnected */ }
    }
    synthsRef.current = []
    if (reverbRef.current) {
      try { reverbRef.current.dispose() } catch {}
      reverbRef.current = null
    }
  }

  async function play(fromMeasure = 1) {
    const { Tone, Midi } = await libs()
    cleanup(Tone)
    await Tone.start()
    setLoading(true)

    Tone.getDestination().volume.value = toDb(volume)

    const b64 = getCurrentMidi()
    if (!b64) { setLoading(false); setPlaying(false); return }

    const bin = atob(b64)
    const buf = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
    const midi = new Midi(buf.buffer)

    let offset = 0
    if (fromMeasure > 1) {
      const times = getMeasureStartTimes()
      offset = times[fromMeasure] ?? 0
    }

    // Fresh reverb each play
    const reverb = new Tone.Reverb({ decay: 1.8, preDelay: 0.01, wet: 0.2 })
    await reverb.generate()
    reverb.toDestination()
    reverbRef.current = reverb

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const noted = midi.tracks.filter((t: any) => t.notes.length > 0)
    // Play only the selected staves (track order matches staff order); none selected = all
    const parts = useStore.getState().selection?.parts
    const picked = parts?.length ? noted.filter((_: unknown, i: number) => parts.includes(i)) : noted
    const activeTracks = picked.length ? picked : noted

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const samplers = activeTracks.map((track: any) =>
      loadSampler(gmName(track.instrument?.number ?? 0), Tone)
    )

    // Connect unique samplers to reverb (multiple tracks may share one sampler)
    const connected = new Set<AnyNode>()
    for (const s of samplers) {
      if (!connected.has(s)) { s.connect(reverb); connected.add(s) }
    }
    synthsRef.current = samplers

    // Wait for all pending AudioBuffers to decode (fast after cache is warm)
    await (Tone as AnyNode).loaded()
    setLoading(false)

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    activeTracks.forEach((track: any, i: number) => {
      const sampler = samplers[i]
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      for (const note of track.notes) {
        const t = note.time - offset
        if (t < -0.01) continue
        Tone.getTransport().schedule((audioTime: number) => {
          sampler.triggerAttackRelease(note.name, note.duration, audioTime, note.velocity)
        }, Math.max(0, t))
      }
    })

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const end = Math.max(0, ...activeTracks.flatMap((t: any) => t.notes.map((n: any) => n.time + n.duration))) - offset
    if (end > 0) {
      Tone.getTransport().schedule(() => stop(), end + 0.3)
    }

    Tone.getTransport().start()
    setPlaying(true)
  }

  function stop() {
    if (toneCache) cleanup(toneCache)
    setPlaying(false)
    setLoading(false)
  }

  return { playing, loading, play, stop, volume, setVolume }
}
