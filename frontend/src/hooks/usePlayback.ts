import { useState, useRef } from 'react'
import { getCurrentMidi, getMeasureStartTimes } from './useVerovio'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyPolySynth = any

let toneCache: typeof import('tone') | null = null
let midiCache: typeof import('@tonejs/midi') | null = null

async function libs() {
  if (!toneCache || !midiCache) {
    ;[toneCache, midiCache] = await Promise.all([import('tone'), import('@tonejs/midi')])
  }
  return { Tone: toneCache!, Midi: midiCache!.Midi }
}

export function usePlayback() {
  const [playing, setPlaying] = useState(false)
  const synthsRef = useRef<AnyPolySynth[]>([])

  function cleanup(Tone: typeof import('tone')) {
    Tone.getTransport().stop()
    Tone.getTransport().cancel()
    for (const s of synthsRef.current) {
      try { s.releaseAll(); s.dispose() } catch { /* already disposed */ }
    }
    synthsRef.current = []
  }

  async function play(fromMeasure = 1) {
    const { Tone, Midi } = await libs()
    cleanup(Tone)
    await Tone.start()

    const b64 = getCurrentMidi()
    if (!b64) { setPlaying(false); return }

    // Decode base64 → Uint8Array
    const bin = atob(b64)
    const buf = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
    const midi = new Midi(buf.buffer)

    // Start-time offset for fromMeasure > 1
    let offset = 0
    if (fromMeasure > 1) {
      const times = getMeasureStartTimes()
      offset = times[fromMeasure] ?? 0
    }

    // One PolySynth per active track
    const activeTracks = midi.tracks.filter((t) => t.notes.length > 0)
    synthsRef.current = activeTracks.map(() => {
      const s = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' as const },
        envelope: { attack: 0.02, decay: 0.1, sustain: 0.5, release: 0.8 },
      })
      s.maxPolyphony = 64
      return s.toDestination()
    })

    // Schedule notes
    activeTracks.forEach((track, i) => {
      const synth = synthsRef.current[i]
      for (const note of track.notes) {
        const t = note.time - offset
        if (t < -0.01) continue
        Tone.getTransport().schedule((audioTime: number) => {
          synth.triggerAttackRelease(note.name, note.duration, audioTime, note.velocity)
        }, Math.max(0, t))
      }
    })

    // Auto-stop after the last note
    const end = Math.max(0, ...activeTracks.flatMap((t) => t.notes.map((n) => n.time + n.duration))) - offset
    if (end > 0) {
      Tone.getTransport().schedule(() => stop(), end + 0.3)
    }

    Tone.getTransport().start()
    setPlaying(true)
  }

  function stop() {
    if (toneCache) cleanup(toneCache)
    setPlaying(false)
  }

  return { playing, play, stop }
}
