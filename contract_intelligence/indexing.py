"""Lexical (TF-IDF, BM25) and Dense (Sentence-Transformer + FAISS) indexing with disk caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "NO")
os.environ.setdefault("USE_TORCH", "YES")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from contract_intelligence.chunking import ContractChunk, chunk_corpus
from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import ContractDocument

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*", re.IGNORECASE)

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "by", "with", "is", "are",
    "this", "that", "under", "between", "and", "or", "any", "what", "how", "does",
    "can", "be", "there", "consider", "contract", "agreement", "if", "so", "who",
    "when", "where", "which", "from", "as", "at",
}

# Deterministic legal terminology expansions for standard contractual clause inquiries
LEGAL_CLAUSE_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"expiration date|expire|term of", re.I),
        "term effective date expire expiration shall continue in full force and effect until years initial term",
    ),
    (
        re.compile(r"governing law|governed by", re.I),
        "governed by and construed in accordance with the laws of the state of jurisdiction venue courts without regard to conflict of laws",
    ),
    (
        re.compile(r"renewal term|renew", re.I),
        "renew renewal automatically extended successive periods additional term year period unless terminated",
    ),
    (
        re.compile(r"notice period.*terminate.*renewal|non-renewal", re.I),
        "prior written notice days months prior to the expiration of the initial term or any renewal term",
    ),
    (
        re.compile(r"terminated? for convenience", re.I),
        "terminate without cause for convenience at any time upon prior written notice days",
    ),
    (
        re.compile(r"anti-assignment|assign", re.I),
        "assign assignment transfer delegate sublicense without the prior written consent of the other party",
    ),
    (
        re.compile(r"cap on liability|limitation of liability", re.I),
        "limitation of liability aggregate liability shall not exceed consequential indirect special punitive damages",
    ),
    (
        re.compile(r"right of first refusal|first offer|first negotiation", re.I),
        "right of first refusal right of first offer first negotiation notice election match third party offer",
    ),
    (
        re.compile(r"competitive restrictions|non-compet|exceptions to competitive", re.I),
        "non-competition compete competitive restriction provided however nothing herein shall prohibit exception",
    ),
    (
        re.compile(r"exclusivity|exclusive", re.I),
        "exclusive exclusivity sole and exclusive territory field of use shall not directly or indirectly",
    ),
    (
        re.compile(r"licenses are granted|license grant", re.I),
        "hereby grants to licensee a non-exclusive exclusive worldwide royalty-free license right to use",
    ),
    (
        re.compile(r"audit rights|inspect", re.I),
        "audit inspect examination books and records accounting certified public accountant retain",
    ),
    (
        re.compile(r"revenue or profit-sharing|royalty", re.I),
        "revenue share profit sharing royalty percentage net sales gross receipts fee payment",
    ),
    (
        re.compile(r"change of control", re.I),
        "change of control merger acquisition consolidation sale of substantially all assets voting securities",
    ),
    (
        re.compile(r"intellectual property ownership|derivative works", re.I),
        "intellectual property ownership title inventions patents copyrights derivative works hereby assigns",
    ),
    (
        re.compile(r"solicitation of employees|non-solicit", re.I),
        "solicit hire employ induce any employee officer consultant non-solicitation",
    ),
    (
        re.compile(r"most favored nation", re.I),
        "most favored nation more favorable terms prices rates offered to any third party",
    ),
    (
        re.compile(r"affiliates.*licensing rights", re.I),
        "affiliates subsidiaries sublicensees right and license extends to affiliates",
    ),
    (
        re.compile(r"minimum commitment|minimum purchase|minimum order", re.I),
        "minimum purchase commitment annual minimum volume quantity quota",
    ),
    (
        re.compile(r"warranty duration|warranty period", re.I),
        "warranty period warrants for a period of days months years from delivery",
    ),
    (
        re.compile(r"insurance", re.I),
        "insurance policy commercial general liability coverage certificate of insurance limits",
    ),
    (
        re.compile(r"covenant not to sue|contest.*validity", re.I),
        "covenant not to sue challenge contest validity enforceability patent trademark",
    ),
    (
        re.compile(r"post-termination|survive|survival", re.I),
        "upon termination expiration survive post-termination wind-down sell-off period",
    ),
]

_EMBEDDING_MODELS: dict[str, Any] = {}
_CROSS_ENCODER_MODELS: dict[str, Any] = {}
_QUERY_VEC_CACHE: dict[tuple[str, str], np.ndarray] = {}
_QUERY_CACHE_LOADED: set[str] = set()


def _query_cache_path(model_name: str, cache_dir: str | Path = "outputs/cache") -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"query_vecs_{slug}.pkl"


def get_embedding_model(model_name: str) -> Any:
    """Lazy-load a cached SentenceTransformer model only when needed."""
    if model_name not in _EMBEDDING_MODELS:
        from sentence_transformers import SentenceTransformer

        try:
            _EMBEDDING_MODELS[model_name] = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _EMBEDDING_MODELS[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODELS[model_name]


def encode_query_texts(
    model_name: str,
    texts: list[str],
    cache_dir: str | Path = "outputs/cache",
) -> np.ndarray:
    """Encode query strings with persistent disk + memory caching so cached queries never initialize the model."""
    c_path = _query_cache_path(model_name, cache_dir)
    if model_name not in _QUERY_CACHE_LOADED and c_path.exists():
        try:
            with open(c_path, "rb") as f:
                saved: dict[str, np.ndarray] = pickle.load(f)
            for k, v in saved.items():
                _QUERY_VEC_CACHE[(model_name, k)] = v
        except Exception:
            pass
        _QUERY_CACHE_LOADED.add(model_name)

    missing = list(dict.fromkeys(t for t in texts if (model_name, t) not in _QUERY_VEC_CACHE))
    if missing:
        model = get_embedding_model(model_name)
        vecs = model.encode(
            missing,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        for t, v in zip(missing, vecs):
            _QUERY_VEC_CACHE[(model_name, t)] = v

        # Persist updated query vectors to disk
        to_save = {k: v for (m, k), v in _QUERY_VEC_CACHE.items() if m == model_name}
        try:
            with open(c_path, "wb") as f:
                pickle.dump(to_save, f)
        except Exception:
            pass

    return np.stack([_QUERY_VEC_CACHE[(model_name, t)] for t in texts], axis=0)


def get_cross_encoder_model(model_name: str) -> Any:
    """Lazy-load a cached CrossEncoder model only when needed."""
    if model_name not in _CROSS_ENCODER_MODELS:
        from sentence_transformers import CrossEncoder

        try:
            _CROSS_ENCODER_MODELS[model_name] = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            _CROSS_ENCODER_MODELS[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_MODELS[model_name]


def legal_tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize legal text while preserving section numbers, decimals, and terms."""
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def decompose_query(query: str, expand_legal: bool = False) -> tuple[str, str, str]:
    """Split a query into (contract_context_part, clause_question_part, expanded_clause_part)."""
    q_clean = query.strip()
    if ";" in q_clean:
        left, right = q_clean.split(";", 1)
        contract_part = re.sub(r"^\s*consider\s+the\s+", "", left.strip(), flags=re.I)
        question_part = right.strip()
    else:
        contract_part = q_clean
        question_part = q_clean

    expanded_question = question_part
    if expand_legal:
        expansions: list[str] = []
        for pattern, extra_terms in LEGAL_CLAUSE_EXPANSIONS:
            if pattern.search(question_part):
                expansions.append(extra_terms)
        if expansions:
            expanded_question = question_part + " " + " ".join(expansions)

    return contract_part, question_part, expanded_question


@dataclass
class BuiltIndex:
    """Container holding all built lexical and dense index structures for a PipelineConfig."""

    config_fingerprint: str
    chunks: list[ContractChunk]
    doc_ids: list[str]
    doc_headers: dict[str, str]
    doc_header_tokens: dict[str, set[str]]

    # TF-IDF over index_text and raw_text
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: Any
    raw_tfidf_vectorizer: TfidfVectorizer
    raw_tfidf_matrix: Any
    doc_tfidf_vectorizer: TfidfVectorizer
    doc_tfidf_matrix: Any

    # BM25 over index_text and raw_text
    bm25_index: BM25Okapi | None
    raw_bm25_index: BM25Okapi | None

    # Dense FAISS index and embeddings
    faiss_index: Any | None
    chunk_embeddings: np.ndarray | None

    build_time_seconds: float
    index_size_bytes: int


def _dense_cache_key(chunks: list[ContractChunk], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for ch in chunks:
        h.update(ch.chunk_id.encode("utf-8"))
        h.update(ch.raw_text.encode("utf-8"))
    return h.hexdigest()[:20]


def build_index(
    contracts: dict[str, ContractDocument],
    config: PipelineConfig,
    use_cache: bool = True,
) -> BuiltIndex:
    """Chunk the indexed contracts and construct lexical and/or dense indices."""
    t0 = time.perf_counter()
    fp = config.index_fingerprint()
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_corpus(contracts, config.chunking, config.legal_context)
    if not chunks:
        raise ValueError("Chunking produced zero chunks from the contract corpus.")

    index_texts = [c.index_text for c in chunks]
    raw_texts = [c.raw_text for c in chunks]

    doc_ids = sorted(contracts.keys())
    doc_headers = {
        did: f"{Path(did).stem.replace('_', ' ').replace('-', ' ')} {contracts[did].text[:2500]}"
        for did in doc_ids
    }
    doc_header_tokens = {
        did: set(legal_tokenize(doc_headers[did], remove_stopwords=True))
        for did in doc_ids
    }

    tfidf_vec = TfidfVectorizer(
        ngram_range=(config.retrieval.tfidf_ngram_min, config.retrieval.tfidf_ngram_max),
        sublinear_tf=config.retrieval.tfidf_sublinear_tf,
        max_features=config.retrieval.tfidf_max_features,
        stop_words="english",
    )
    tfidf_mat = tfidf_vec.fit_transform(index_texts)

    raw_tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=25000,
        stop_words="english",
    )
    raw_tfidf_mat = raw_tfidf_vec.fit_transform(raw_texts)

    doc_tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    doc_tfidf_mat = doc_tfidf_vec.fit_transform([doc_headers[did] for did in doc_ids])

    bm25_idx: BM25Okapi | None = None
    raw_bm25_idx: BM25Okapi | None = None
    if (
        config.retrieval.strategy in ("bm25", "hybrid")
        or config.abstention.mode == "dual_gate"
        or config.legal_context.doc_routing_weight > 0
    ):
        tokenized_index = [legal_tokenize(t, remove_stopwords=True) for t in index_texts]
        bm25_idx = BM25Okapi(
            tokenized_index,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
        )
        tokenized_raw = [legal_tokenize(t, remove_stopwords=True) for t in raw_texts]
        raw_bm25_idx = BM25Okapi(
            tokenized_raw,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
        )

    faiss_idx: Any | None = None
    embeddings: np.ndarray | None = None
    if config.retrieval.strategy in ("dense", "hybrid"):
        d_key = _dense_cache_key(chunks, config.retrieval.dense_model_name)
        emb_file = cache_dir / f"dense_{d_key}.npy"
        meta_file = cache_dir / f"dense_{d_key}.json"
        if use_cache and emb_file.exists() and meta_file.exists():
            embeddings = np.load(emb_file)
        else:
            model = get_embedding_model(config.retrieval.dense_model_name)
            dense_inputs = [
                (f"{c.section_heading}: {c.raw_text}" if c.section_heading else c.raw_text)
                for c in chunks
            ]
            embeddings = model.encode(
                dense_inputs,
                batch_size=config.retrieval.dense_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            if use_cache:
                np.save(emb_file, embeddings)
                meta_file.write_text(
                    json.dumps(
                        {
                            "model_name": config.retrieval.dense_model_name,
                            "num_chunks": len(chunks),
                            "config_fingerprint": fp,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        dim = int(embeddings.shape[1])
        faiss.omp_set_num_threads(1)
        faiss_idx = faiss.IndexFlatIP(dim)
        faiss_idx.add(embeddings)

    elapsed = time.perf_counter() - t0

    size_bytes = int(tfidf_mat.data.nbytes + tfidf_mat.indices.nbytes + tfidf_mat.indptr.nbytes)
    if embeddings is not None:
        size_bytes += int(embeddings.nbytes)
    if bm25_idx is not None:
        size_bytes += len(pickle.dumps(bm25_idx.doc_freqs))

    return BuiltIndex(
        config_fingerprint=fp,
        chunks=chunks,
        doc_ids=doc_ids,
        doc_headers=doc_headers,
        doc_header_tokens=doc_header_tokens,
        tfidf_vectorizer=tfidf_vec,
        tfidf_matrix=tfidf_mat,
        raw_tfidf_vectorizer=raw_tfidf_vec,
        raw_tfidf_matrix=raw_tfidf_vec.transform(raw_texts) if raw_tfidf_mat is None else raw_tfidf_mat,
        doc_tfidf_vectorizer=doc_tfidf_vec,
        doc_tfidf_matrix=doc_tfidf_mat,
        bm25_index=bm25_idx,
        raw_bm25_index=raw_bm25_idx,
        faiss_index=faiss_idx,
        chunk_embeddings=embeddings,
        build_time_seconds=elapsed,
        index_size_bytes=size_bytes,
    )
