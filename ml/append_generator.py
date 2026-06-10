"""Inference pipeline for appending new measures to an existing score."""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Generator

from ml.tokenizer import (
    score_to_tokens_by_measure, score_to_header_tokens, score_to_xml_str,
    tokens_to_score, BOS, EOS, BAR,
)
from ml.model import load_model

OUTPUT_DIR = Path("output")

def append_and_export(
    xml_str: str,
    num_bars: int = 4,
    max_new: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = "GEN-D1024-H8-L8",
) -> dict[str, str]:
    """Append num_bars newly generated measures to an existing score.

    Tokenizes the existing score, uses it as a context prefix via
    generate_standard(), extracts the newly generated measures, and appends
    them to the original score XML.

    Returns:
        Dict with path to output file: {"xml": ...}
    """
    import music21 as m21

    log("Parsing existing score...")
    orig_score = m21.converter.parseData(xml_str, format="musicxml")

    log("Loading generation model...")
    model, cfg = load_model(model_name, device)
    max_seq_len: int = cfg.get("max_seq_len", 2048)
    log(f"Model loaded ({model.num_params():,} params, max_seq_len={max_seq_len})")

    header_tokens = score_to_header_tokens(orig_score)
    measure_tokens = score_to_tokens_by_measure(orig_score)

    # Flatten measure tokens in order
    flat_measures: list[int] = []
    for m_num in sorted(measure_tokens.keys()):
        flat_measures.extend(measure_tokens[m_num])

    if not flat_measures:
        flat_measures = [BAR]

    # Truncate context from the left if it would exceed the model window.
    # Reserve 256 tokens for new generation.
    max_ctx = max_seq_len - 256
    header_overhead = 1 + len(header_tokens)  # BOS + header
    if header_overhead + len(flat_measures) > max_ctx:
        excess = header_overhead + len(flat_measures) - max_ctx
        # Advance past `excess` tokens, then align to the next BAR boundary
        i = excess
        while i < len(flat_measures) and flat_measures[i] != BAR:
            i += 1
        flat_measures = flat_measures[i:]
        log(f"Context truncated: using last {flat_measures.count(BAR)} of {len(measure_tokens)} measures")

    n_ctx_bars = flat_measures.count(BAR)
    target_bars = n_ctx_bars + num_bars
    log(f"Generating {num_bars} new bars (context has {n_ctx_bars} bars, target={target_bars})...")

    generated = model.generate_standard(
        header_tokens=header_tokens,
        first_bar_prefix=flat_measures,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    # Reconstruct the continuation score (context measures + new measures)
    full_tokens = [BOS] + header_tokens + flat_measures + generated + [EOS]
    continuation_score = tokens_to_score(full_tokens)

    # Extract only the new measures (those after the n_ctx_bars context measures)
    orig_parts = list(orig_score.parts)
    cont_parts = list(continuation_score.parts)

    # Determine the highest measure number in the original score for renumbering
    if orig_parts:
        orig_last_m_num = max(
            (m.number or 0)
            for p in orig_parts
            for m in p.getElementsByClass(m21.stream.Measure)
        )
    else:
        orig_last_m_num = 0

    for part_idx, orig_part in enumerate(orig_parts):
        next_m_num = orig_last_m_num + 1
        if part_idx < len(cont_parts):
            cont_measures = list(cont_parts[part_idx].getElementsByClass(m21.stream.Measure))
            new_measures = cont_measures[n_ctx_bars:n_ctx_bars + num_bars]
        else:
            new_measures = []
        for new_m in new_measures:
            new_m.number = next_m_num
            orig_part.append(new_m)
            next_m_num += 1
        for _ in range(len(new_measures), num_bars):
            rest_m = m21.stream.Measure(number=next_m_num)
            rest_m.append(m21.note.Rest(quarterLength=4.0))
            orig_part.append(rest_m)
            next_m_num += 1

    try:
        for part in orig_score.parts:
            part.makeBeams(inPlace=True)
    except Exception as e:
        log(f"makeBeams skipped: {e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    xml_path = str(OUTPUT_DIR / "appended.xml")
    try:
        orig_score.write("musicxml", fp=xml_path)
        log(f"Saved appended MusicXML -> {xml_path}")
    except Exception as e:
        log(f"MusicXML export failed: {e}")
        raise

    total_measures = orig_last_m_num + num_bars
    log(f"Score now has {total_measures} measures ({num_bars} appended)")
    return {"xml": xml_path}


def append_and_stream(
    xml_str: str,
    num_bars: int = 4,
    max_new: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = "GEN-D1024-H8-L8",
) -> Generator[tuple[int, str, int], None, None]:
    """Stream partial MusicXML as each new measure is generated and merged.

    Yields (bar_index, xml_str, total_bars) where bar_index is the count of
    new measures appended so far.
    """
    import music21 as m21
    import copy

    log("Parsing existing score...")
    orig_score = m21.converter.parseData(xml_str, format="musicxml")

    log("Loading generation model...")
    model, cfg = load_model(model_name, device)
    max_seq_len: int = cfg.get("max_seq_len", 2048)
    log(f"Model loaded ({model.num_params():,} params)")

    header_tokens = score_to_header_tokens(orig_score)
    measure_tokens = score_to_tokens_by_measure(orig_score)

    flat_measures: list[int] = []
    for m_num in sorted(measure_tokens.keys()):
        flat_measures.extend(measure_tokens[m_num])

    if not flat_measures:
        flat_measures = [BAR]

    max_ctx = max_seq_len - 256
    header_overhead = 1 + len(header_tokens)
    if header_overhead + len(flat_measures) > max_ctx:
        excess = header_overhead + len(flat_measures) - max_ctx
        i = excess
        while i < len(flat_measures) and flat_measures[i] != BAR:
            i += 1
        flat_measures = flat_measures[i:]
        log(f"Context truncated: using last {flat_measures.count(BAR)} of {len(measure_tokens)} measures")

    n_ctx_bars = flat_measures.count(BAR)
    target_bars = n_ctx_bars + num_bars

    orig_parts = list(orig_score.parts)
    orig_last_m_num = max(
        (m.number or 0)
        for p in orig_parts
        for m in p.getElementsByClass(m21.stream.Measure)
    ) if orig_parts else 0

    last_snapshot: list[int] = []

    for snapshot in model.generate_standard_stream(
        header_tokens=header_tokens,
        first_bar_prefix=flat_measures,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
    ):
        last_snapshot = snapshot
        new_bars = snapshot.count(BAR)

        full_tokens = [BOS] + header_tokens + flat_measures + snapshot + [EOS]
        try:
            continuation_score = tokens_to_score(full_tokens)
            cont_parts = list(continuation_score.parts)

            partial_score = copy.deepcopy(orig_score)
            partial_orig_parts = list(partial_score.parts)

            for part_idx, orig_part in enumerate(partial_orig_parts):
                curr_m_num = orig_last_m_num + 1
                if part_idx < len(cont_parts):
                    cont_measures = list(cont_parts[part_idx].getElementsByClass(m21.stream.Measure))
                    new_measures = cont_measures[n_ctx_bars:n_ctx_bars + new_bars]
                else:
                    new_measures = []
                for i, new_m in enumerate(new_measures):
                    new_m.number = curr_m_num + i
                    orig_part.append(new_m)
                for i in range(len(new_measures), new_bars):
                    rest_m = m21.stream.Measure(number=curr_m_num + i)
                    rest_m.append(m21.note.Rest(quarterLength=4.0))
                    orig_part.append(rest_m)

            xml_out = score_to_xml_str(partial_score)
            # log(f"Streamed {new_bars} new bar(s) of {num_bars}")
            yield (new_bars, xml_out, num_bars)
        except Exception as e:
            log(f"Append stream XML error at {new_bars} bars: {e}")

    # Final emit with complete last measure
    full_tokens = [BOS] + header_tokens + flat_measures + last_snapshot + [EOS]
    try:
        continuation_score = tokens_to_score(full_tokens)
        cont_parts = list(continuation_score.parts)

        final_score = copy.deepcopy(orig_score)
        final_orig_parts = list(final_score.parts)

        total_new = last_snapshot.count(BAR)
        for part_idx, orig_part in enumerate(final_orig_parts):
            curr_m_num = orig_last_m_num + 1
            if part_idx < len(cont_parts):
                cont_measures = list(cont_parts[part_idx].getElementsByClass(m21.stream.Measure))
                new_measures = cont_measures[n_ctx_bars:n_ctx_bars + total_new]
            else:
                new_measures = []
            for i, new_m in enumerate(new_measures[:num_bars]):
                new_m.number = curr_m_num + i
                orig_part.append(new_m)
            for i in range(min(len(new_measures), num_bars), num_bars):
                rest_m = m21.stream.Measure(number=curr_m_num + i)
                rest_m.append(m21.note.Rest(quarterLength=4.0))
                orig_part.append(rest_m)

        try:
            for part in final_score.parts:
                part.makeBeams(inPlace=True)
        except Exception:
            pass

        xml_out = score_to_xml_str(final_score)
        log(f"Append stream complete: {total_new} new bars")
        yield (total_new, xml_out, num_bars)
    except Exception as e:
        log(f"Final append stream XML error: {e}")
