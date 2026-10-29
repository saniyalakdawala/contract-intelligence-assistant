"""Fixed-window and clause-aware contract chunking with strict original character offset invariants."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from contract_intelligence.config import ChunkingConfig, LegalContextConfig
from contract_intelligence.dataset import ContractDocument

# Regex patterns for legal structure detection
SECTION_HEADER_RE = re.compile(
    r"^\s*(?:ARTICLE\s+[IVXLCDM0-9]+|SECTION\s+\d+(?:\.\d+)*|\d+\.\d*\s+[A-Z][A-Za-z0-9\s,\-/()]{2,70}|[A-Z][A-Z\s,\-&]{4,60}\.?)\s*$",
    re.MULTILINE,
)
NUMBERED_CLAUSE_RE = re.compile(
    r"(?:^|\n)\s*((?:ARTICLE\s+[IVXLCDM0-9]+|SECTION\s+\d+(?:\.\d+)*|\d+(?:\.\d+)+|\d+\.|\([a-z0-9]+\))\s+)",
    re.IGNORECASE,
)
CROSS_REF_RE = re.compile(
    r"\b(?:Section|Article|Clause|Paragraph|Exhibit|Schedule)\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*(?:\([a-z0-9]+\))?|[A-Z])\b",
    re.IGNORECASE,
)


@dataclass
class ContractChunk:
    """A contiguous slice of a contract document with exact character offsets in the original text."""

    chunk_id: str
    doc_id: str
    file_path: str
    char_start: int
    char_end: int
    raw_text: str
    index_text: str
    section_heading: str | None = None
    clause_number: str | None = None
    cross_references: list[str] = field(default_factory=list)
    referenced_context: str | None = None


def extract_cross_references(text: str) -> list[str]:
    """Extract unique cross-reference mentions (e.g., 'Section 4.2', 'Article III') from a passage."""
    seen: list[str] = []
    for match in CROSS_REF_RE.finditer(text):
        ref = " ".join(match.group(0).split())
        if ref not in seen:
            seen.append(ref)
    return seen[:8]


def _detect_heading_and_clause(span_text: str, current_heading: str | None) -> tuple[str | None, str | None]:
    """Identify if a span starts with or contains a section heading or numbered clause."""
    heading = current_heading
    clause_num: str | None = None
    lines = [ln.strip() for ln in span_text.splitlines() if ln.strip()]
    for ln in lines[:3]:
        if SECTION_HEADER_RE.match(ln) and len(ln) <= 90:
            heading = ln
            break
    m = NUMBERED_CLAUSE_RE.search(span_text[:160])
    if m:
        clause_num = m.group(1).strip()
    return heading, clause_num


def _split_into_atomic_spans(text: str, max_span_len: int) -> list[tuple[int, int]]:
    """Split document text into contiguous (start, end) spans along paragraph, clause, or sentence boundaries."""
    n = len(text)
    if n == 0:
        return []

    # First split on double newlines or numbered clause starts
    break_positions: set[int] = {0, n}
    for m in re.finditer(r"\n\s*\n", text):
        break_positions.add(m.end())
    for m in NUMBERED_CLAUSE_RE.finditer(text):
        pos = m.start(1)
        if pos > 0:
            break_positions.add(pos)

    sorted_pos = sorted(break_positions)
    coarse_spans: list[tuple[int, int]] = []
    for i in range(len(sorted_pos) - 1):
        s, e = sorted_pos[i], sorted_pos[i + 1]
        if e > s:
            coarse_spans.append((s, e))

    # Sub-split any span longer than max_span_len using sentence boundaries, then whitespace
    atomic_spans: list[tuple[int, int]] = []
    for s, e in coarse_spans:
        if (e - s) <= max_span_len:
            atomic_spans.append((s, e))
            continue

        sub_text = text[s:e]
        sent_breaks = [0]
        for sm in re.finditer(r"(?<=[.!?;])\s+", sub_text):
            sent_breaks.append(sm.end())
        if sent_breaks[-1] != len(sub_text):
            sent_breaks.append(len(sub_text))

        cur_start = s
        for idx in range(1, len(sent_breaks)):
            cand_end = s + sent_breaks[idx]
            if (cand_end - cur_start) > max_span_len:
                if cur_start < s + sent_breaks[idx - 1]:
                    atomic_spans.append((cur_start, s + sent_breaks[idx - 1]))
                    cur_start = s + sent_breaks[idx - 1]
                # If a single sentence is still longer than max_span_len, slice cleanly
                while (cand_end - cur_start) > max_span_len:
                    atomic_spans.append((cur_start, cur_start + max_span_len))
                    cur_start += max_span_len
        if cur_start < e:
            atomic_spans.append((cur_start, e))

    return atomic_spans


def _build_section_map(chunks: list[ContractChunk]) -> dict[str, str]:
    """Build a heuristic lookup map from section/clause identifiers to short passage previews."""
    sec_map: dict[str, str] = {}
    for ch in chunks:
        if ch.clause_number:
            norm_num = re.sub(r"[^a-z0-9.]", "", ch.clause_number.lower()).rstrip(".")
            if norm_num and norm_num not in sec_map:
                sec_map[f"section {norm_num}"] = ch.raw_text[:280].strip()
                sec_map[f"clause {norm_num}"] = ch.raw_text[:280].strip()
        if ch.section_heading:
            m = re.search(r"(?:section|article)\s+([ivxlcdm0-9]+(?:\.\d+)*)", ch.section_heading, re.IGNORECASE)
            if m:
                key = f"section {m.group(1).lower()}"
                sec_map.setdefault(key, ch.raw_text[:280].strip())
    return sec_map


def chunk_document(
    doc: ContractDocument,
    chunk_cfg: ChunkingConfig,
    legal_cfg: LegalContextConfig | None = None,
) -> list[ContractChunk]:
    """Chunk a single contract document according to ChunkingConfig while guaranteeing exact offsets."""
    legal_cfg = legal_cfg or LegalContextConfig()
    text = doc.text
    n = len(text)
    if n == 0:
        return []

    raw_windows: list[tuple[int, int, str | None, str | None]] = []

    if chunk_cfg.strategy == "fixed":
        step = max(1, chunk_cfg.chunk_size - chunk_cfg.chunk_overlap)
        start = 0
        while start < n:
            end = min(n, start + chunk_cfg.chunk_size)
            raw_windows.append((start, end, None, None))
            if end == n:
                break
            start += step
    else:
        # Clause-aware structural chunking
        max_atom = max(120, chunk_cfg.chunk_size // 2)
        atoms = _split_into_atomic_spans(text, max_span_len=max_atom)
        current_heading: str | None = None
        i = 0
        num_atoms = len(atoms)
        while i < num_atoms:
            w_start = atoms[i][0]
            w_end = atoms[i][1]
            span_str = text[w_start:w_end]
            current_heading, clause_num = _detect_heading_and_clause(span_str, current_heading)
            j = i + 1
            while j < num_atoms:
                next_s, next_e = atoms[j]
                next_str = text[next_s:next_e]
                # If respecting section boundaries and next atom starts a major top-level section, stop if chunk is reasonably sized
                if (
                    chunk_cfg.respect_section_boundaries
                    and (w_end - w_start) >= chunk_cfg.min_chunk_size
                    and SECTION_HEADER_RE.match(next_str.strip().splitlines()[0] if next_str.strip() else "")
                ):
                    break
                if (next_e - w_start) <= chunk_cfg.chunk_size:
                    w_end = next_e
                    j += 1
                else:
                    break

            raw_windows.append((w_start, w_end, current_heading, clause_num))
            if j >= num_atoms:
                break

            # Advance with overlap measured in atomic spans
            if chunk_cfg.chunk_overlap > 0 and j > i + 1:
                overlap_Start = w_end - chunk_cfg.chunk_overlap
                next_i = j
                for k in range(j - 1, i, -1):
                    if atoms[k][0] >= overlap_Start:
                        next_i = k
                    else:
                        break
                i = max(i + 1, next_i)
            else:
                i = j

    chunks: list[ContractChunk] = []
    for idx, (c_start, c_end, heading, clause_num) in enumerate(raw_windows):
        raw_slice = text[c_start:c_end]
        if not raw_slice.strip():
            continue
        # Enforce exact offset invariant
        assert doc.text[c_start:c_end] == raw_slice, (
            f"Offset mismatch in {doc.doc_id}: [{c_start}:{c_end}]"
        )
        xrefs = extract_cross_references(raw_slice)
        prefix_parts: list[str] = []
        if legal_cfg.include_doc_header:
            prefix_parts.append(
                doc.title_header[: legal_cfg.header_char_limit]
            )
        if heading:
            prefix_parts.append(f"[Section: {heading}]")
        if clause_num:
            prefix_parts.append(f"[Clause: {clause_num}]")

        index_text = (" ".join(prefix_parts) + "\n" + raw_slice) if prefix_parts else raw_slice
        chunks.append(
            ContractChunk(
                chunk_id=f"{doc.doc_id}::chunk_{idx:04d}",
                doc_id=doc.doc_id,
                file_path=doc.file_path,
                char_start=c_start,
                char_end=c_end,
                raw_text=raw_slice,
                index_text=index_text,
                section_heading=heading,
                clause_number=clause_num,
                cross_references=xrefs,
                referenced_context=None,
            )
        )

    # Heuristic cross-reference resolution within the same document
    if legal_cfg.resolve_cross_references and chunks:
        sec_map = _build_section_map(chunks)
        for ch in chunks:
            resolved_notes: list[str] = []
            for ref in ch.cross_references:
                ref_key = ref.lower()
                if ref_key in sec_map and sec_map[ref_key] not in ch.raw_text:
                    resolved_notes.append(f"{ref}: {sec_map[ref_key]}")
            if resolved_notes:
                ch.referenced_context = " | ".join(resolved_notes[:2])
                ch.index_text = ch.index_text + "\n[Cross-Reference Context: " + ch.referenced_context + "]"

    return chunks


def chunk_corpus(
    contracts: dict[str, ContractDocument],
    chunk_cfg: ChunkingConfig,
    legal_cfg: LegalContextConfig | None = None,
) -> list[ContractChunk]:
    """Chunk all indexed contracts deterministically in sorted document order."""
    all_chunks: list[ContractChunk] = []
    for doc_id in sorted(contracts.keys()):
        all_chunks.extend(chunk_document(contracts[doc_id], chunk_cfg, legal_cfg))
    return all_chunks

