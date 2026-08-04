"""Key spelling helpers shared by the generation and key-signature endpoints."""

from __future__ import annotations


def normalize_tonic(tonic: str) -> str:
    """Frontend spelling to music21's ("Bb" -> "B-", "f#" -> "F#")."""
    t = tonic.strip()
    return t[:1].upper() + t[1:].replace("b", "-").replace("♭", "-").replace("♯", "#")


def key_index(tonic: str | None, mode: str | None) -> int | None:
    """The model's key index (0..12, 6 = C, a fifth per step) for a tonic and mode, letting music21
    do the theory: A minor and C major both come back as 6. None when no tonic was asked for or the
    pair does not name a key, which the model reads as "carry on in the key already in effect"."""
    if not tonic:
        return None
    import music21 as m21

    try:
        sharps = m21.key.Key(normalize_tonic(tonic), (mode or "major").strip().lower()).sharps
    except Exception:
        return None
    # Signatures past 6 accidentals (C# major, Cb major) fold onto their enharmonic index — the
    # model stores only the tonic's pitch class, so the two spellings are the same token.
    return (sharps + 6) % 12
