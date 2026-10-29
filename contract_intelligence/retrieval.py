"""ClauseTrace retrieval engine supporting TF-IDF, BM25, Dense, Hybrid, Reranking, and Abstention."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contract_intelligence.abstention import AbstentionDecision, evaluate_abstention
from contract_intelligence.config import PipelineConfig
from contract_intelligence.dataset import ContractDocument
from contract_intelligence.hybrid import (
    apply_cross_encoder_rerank,
    bm25_saturation_normalize,
    linear_score_fusion,
    reciprocal_rank_fusion,
)
from contract_intelligence.indexing import (
    BuiltIndex,
    build_index,
    decompose_query,
    encode_query_texts,
    legal_tokenize,
)
from contract_intelligence.validation import verify_evidence_offset

FRAMING_STOPWORDS = {
    "what", "is", "are", "the", "of", "for", "in", "under", "this", "contract",
    "agreement", "does", "can", "be", "there", "any", "how", "to", "and", "or",
    "when", "who", "which", "by", "with", "if", "so", "conditions", "between",
    "party", "parties", "one", "event", "happens", "include", "contain", "permit",
    "allow", "required", "exceptions", "preventing", "assigned", "have", "has",
    "clause", "designated", "included", "arrangements",
}

LEGAL_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "expir": ["expir", "term", "until", "effective", "commenc", "year", "20"],
    "date": ["date", "day", "month", "year", "199", "200", "201", "202"],
    "govern": ["govern", "constru", "choice of law"],
    "law": ["govern", "constru", "laws of", "state of", "commonwealth", "province"],
    "termin": ["termin", "cancel", "expir", "discontinu", "cease"],
    "conven": ["conven", "without cause", "any time", "prior written notice"],
    "anti": ["assign", "transfer", "delegat", "prior written consent"],
    "assign": ["assign", "transfer", "delegat", "ownership", "title", "vest", "belong", "hereby"],
    "cap": ["limit", "exceed", "liabil", "consequenti", "damag", "aggreg", "maxim"],
    "liabil": ["liabil", "damag", "exceed", "indemn", "consequenti"],
    "compet": ["compet", "restrict", "prohibit", "solicit", "non-comp"],
    "restrict": ["restrict", "compet", "prohibit", "shall not", "except"],
    "exclus": ["exclus", "sole", "compet", "non-comp", "territor", "grant"],
    "revenu": ["revenu", "profit", "royalt", "commiss", "gross", "net sales", "split", "share", "percent", "%"],
    "profit": ["profit", "revenu", "royalt", "commiss", "gross", "net sales", "split", "share", "percent", "%"],
    "sharin": ["share", "sharing", "royalt", "percent", "%", "split", "commiss"],
    "intell": ["intell", "propert", "inven", "patent", "copyright", "trademark", "owner", "title", "work", "mark", "propriet"],
    "proper": ["propert", "intell", "inven", "patent", "copyright", "trademark", "owner", "title", "work", "mark", "propriet"],
    "owners": ["owner", "title", "vest", "belong", "assign", "sole", "propert", "retain"],
    "audit": ["audit", "inspect", "book", "record", "account", "examin", "verif"],
    "right": ["right", "licens", "grant", "entitl", "option", "refus", "offer", "audit"],
    "renew": ["renew", "extend", "extens", "successiv", "automatic"],
    "notice": ["notice", "notifi", "written", "days", "months", "prior"],
    "period": ["period", "term", "days", "months", "years", "date", "prior", "within"],
    "refus": ["refus", "first offer", "first negoti", "option", "match", "third party"],
    "offer": ["offer", "refus", "negoti", "first", "option", "match"],
    "negoti": ["negoti", "refus", "offer", "first", "option"],
    "favor": ["most favored", "more favorable", "favorable", "equal or better"],
    "nation": ["most favored", "more favorable", "favorable"],
    "contro": ["control", "merger", "acqui", "consolid", "voting", "ownership"],
    "change": ["change of control", "merger", "acqui", "consolid", "control"],
    "solici": ["solici", "hire", "employ", "induc", "recrui", "entic"],
    "employ": ["employ", "solici", "hire", "personn", "staff", "consult", "offic"],
    "affili": ["affili", "subsidiar", "parent", "sublicens"],
    "licens": ["licens", "grant", "right to use", "sublicens", "hereby grants"],
    "grant": ["grant", "licens", "right", "hereby", "use", "author"],
    "transf": ["transfer", "non-transfer", "assign", "sublicens"],
    "non": ["non-", "not ", "never", "prohibit", "restrict", "without"],
    "third": ["third", "beneficiar", "party", "affiliate", "sublicens"],
    "benefi": ["beneficiar", "third party", "third-party", "enforc", "inure"],
    "covena": ["covenant", "sue", "challeng", "contest", "valid", "assert"],
    "sue": ["sue", "covenant", "challeng", "contest", "litigat", "claim"],
    "warran": ["warran", "defect", "conform", "merchant", "repair", "replac"],
    "durati": ["durati", "period", "month", "year", "day", "term", "warran"],
    "insura": ["insura", "polic", "coverag", "underwrit", "liabil", "certific"],
    "minimu": ["minimu", "commit", "quanti", "volum", "purchas", "quota", "annual"],
    "commit": ["commit", "minimu", "oblig", "purchas", "quanti", "volum"],
    "liquid": ["liquidat", "damag", "penalt", "delay"],
    "surviv": ["surviv", "termin", "expir", "remain in effect", "after"],
}

CRITICAL_CONCEPT_GATES: list[tuple[str, list[str]]] = [
    ("renew", ["renew", "extend", "successiv"]),
    ("governing law", ["govern", "constru", "laws of"]),
    ("sublicens", ["sublicens", "sub-licens"]),
    ("derivative", ["deriv", "modif", "improv", "inven", "work"]),
    ("cap on liability", ["liabil", "exceed", "consequenti", "damag"]),
    ("most favored nation", ["favor", "more favorable"]),
    ("audit", ["audit", "inspect", "book", "record", "account"]),
    ("revenue or profit", ["revenu", "profit", "royalt", "commiss", "net sales", "gross", "%", "percent"]),
    ("source code", ["source code", "escrow"]),
    ("cryptocurrency", ["cryptocurr", "bitcoin"]),
    ("gdpr", ["gdpr", "personal data"]),
    ("siac", ["siac", "singapore"]),
    ("ccpa", ["ccpa", "california consumer"]),
    ("reach", ["reach", "european union"]),
    ("liquidated damages", ["liquidat"]),
    ("artificial intelligence", ["artificial intelligence", "voice clone"]),
]

GENERIC_CONTRACT_WORDS = {
    "consider", "contract", "agreement", "between", "and", "the", "for", "of",
    "inc", "llc", "ltd", "corp", "corporation", "company", "plc", "lp", "group",
    "holdings", "international", "technologies", "services", "amendment", "no",
    "service", "sponsorship", "strategic", "alliance", "joint", "venture",
    "co", "branding", "endorsement", "marketing", "supply", "agency", "franchise",
    "transportation", "intellectual", "property", "reseller", "support",
    "maintenance", "promotion", "cooperation", "affiliate", "business",
    "non", "competition", "right", "first", "offer", "wireless", "content",
    "license", "asset", "sale", "termination", "parties", "party", "what",
    "does", "is", "are", "under", "this", "allow", "when", "who", "owns",
}


@dataclass
class EvidencePassage:
    """Structured evidence object anchored to exact character offsets in the original contract."""

    doc_id: str
    file_path: str
    char_start: int
    char_end: int
    passage_text: str
    retrieval_method: str
    rank: int
    retrieval_score: float
    scores: dict[str, float] = field(default_factory=dict)
    section_heading: str | None = None
    clause_number: str | None = None
    cross_references: list[str] = field(default_factory=list)
    referenced_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "passage_text": self.passage_text,
            "retrieval_method": self.retrieval_method,
            "rank": self.rank,
            "retrieval_score": round(float(self.retrieval_score), 4),
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "section_heading": self.section_heading,
            "clause_number": self.clause_number,
            "cross_references": self.cross_references,
            "referenced_context": self.referenced_context,
        }


@dataclass
class RetrievalResponse:
    """Complete response returned by ClauseTrace for a single natural-language query."""

    query: str
    abstained: bool
    abstention: AbstentionDecision
    passages: list[EvidencePassage]
    unfiltered_top_passages: list[EvidencePassage]
    retrieval_method: str
    latency_ms: float
    top_score: float
    top_clause_score: float
    top_doc_match_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "abstention_status": self.abstention.status,
            "abstention_reason": self.abstention.reason,
            "abstention_diagnostics": self.abstention.diagnostics,
            "retrieval_method": self.retrieval_method,
            "latency_ms": round(self.latency_ms, 3),
            "top_score": round(self.top_score, 4),
            "top_clause_score": round(self.top_clause_score, 4),
            "top_doc_match_score": round(self.top_doc_match_score, 4),
            "passages": [p.to_dict() for p in self.passages],
        }


class ClauseTraceRetriever:
    """Modular legal clause retrieval pipeline supporting configurable strategies and honest abstention."""

    def __init__(
        self,
        contracts: dict[str, ContractDocument],
        config: PipelineConfig,
        index: BuiltIndex | None = None,
        use_cache: bool = True,
    ) -> None:
        self.contracts = contracts
        self.config = config
        self.index = index or build_index(contracts, config, use_cache=use_cache)

    def _compute_doc_alignment_scores(self, contract_part: str) -> dict[str, float]:
        """Compute alignment between the query's contract/party description and each indexed document."""
        q_vec = self.index.doc_tfidf_vectorizer.transform([contract_part])
        tfidf_sims = (self.index.doc_tfidf_matrix @ q_vec.T).toarray().ravel()

        q_tokens = set(legal_tokenize(contract_part, remove_stopwords=True))
        distinctive_tokens = {t for t in q_tokens if t not in GENERIC_CONTRACT_WORDS and len(t) >= 3}

        doc_scores: dict[str, float] = {}
        for i, did in enumerate(self.index.doc_ids):
            hdr_tokens = self.index.doc_header_tokens[did]
            if distinctive_tokens:
                overlap = len(distinctive_tokens & hdr_tokens) / float(len(distinctive_tokens))
            else:
                overlap = float(tfidf_sims[i])
            doc_scores[did] = 0.45 * float(tfidf_sims[i]) + 0.55 * float(overlap)
        return doc_scores

    def _compute_concept_gate_score(
        self,
        question_part: str,
        top_passages: list[EvidencePassage],
        top_doc_id: str,
    ) -> float:
        """Compute legal concept & anchor-term verification score against top retrieved passages."""
        if not top_passages:
            return 0.0
        same_doc_passages = [p for p in top_passages[:3] if p.doc_id == top_doc_id] or top_passages[:3]
        combined_raw = " ".join(p.passage_text.lower() for p in same_doc_passages)
        q_low = question_part.lower()

        for trig, required_any in CRITICAL_CONCEPT_GATES:
            if trig in q_low and not any(req in combined_raw for req in required_any):
                return 0.05

        q_toks = [
            t
            for t in legal_tokenize(question_part, remove_stopwords=True)
            if t not in FRAMING_STOPWORDS and len(t) >= 3
        ]
        if not q_toks:
            return 1.0

        matched = 0
        unmatched_specific = 0
        for t in q_toks:
            stem = t[:6]
            if stem in LEGAL_CONCEPT_SYNONYMS:
                if any(s in combined_raw for s in LEGAL_CONCEPT_SYNONYMS[stem]):
                    matched += 1
            else:
                if t[:5] in combined_raw:
                    matched += 1
                else:
                    unmatched_specific += 1

        base_cov = matched / float(len(q_toks))
        if unmatched_specific > 0:
            base_cov *= 0.35**unmatched_specific
        return float(base_cov)

    def _compute_clause_scores(self, question_part: str, expanded_question: str) -> np.ndarray:
        """Compute direct clause relevance scores against raw chunk text (excluding any document header)."""
        q_raw_vec = self.index.raw_tfidf_vectorizer.transform([expanded_question])
        raw_tfidf_scores = (self.index.raw_tfidf_matrix @ q_raw_vec.T).toarray().ravel()

        if self.index.raw_bm25_index is not None:
            q_toks = legal_tokenize(expanded_question, remove_stopwords=True)
            raw_bm25 = np.array(self.index.raw_bm25_index.get_scores(q_toks), dtype=np.float64)
            raw_bm25_cal = bm25_saturation_normalize(raw_bm25, k_sat=6.0)
            return 0.50 * raw_tfidf_scores + 0.50 * raw_bm25_cal

        return raw_tfidf_scores

    def search(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        """Retrieve ranked supporting contract passages with exact offsets or abstain honestly."""
        t0 = time.perf_counter()
        k = top_k if top_k is not None else self.config.retrieval.top_k
        method_label = self.config.retrieval.strategy + (
            "+rerank" if self.config.rerank.enabled else ""
        )

        q_clean = (query or "").strip()
        if len(q_clean) < 3 or not any(ch.isalnum() for ch in q_clean):
            decision = evaluate_abstention(q_clean, 0.0, 0.0, 0.0, self.config.abstention)
            return RetrievalResponse(
                query=query,
                abstained=True,
                abstention=decision,
                passages=[],
                unfiltered_top_passages=[],
                retrieval_method=method_label,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                top_score=0.0,
                top_clause_score=0.0,
                top_doc_match_score=0.0,
            )

        contract_part, question_part, expanded_question = decompose_query(
            q_clean,
            expand_legal=self.config.legal_context.enable_legal_query_expansion,
        )
        use_routed_clause_query = self.config.legal_context.doc_routing_weight > 0.0 and ";" in q_clean
        effective_query = expanded_question if use_routed_clause_query else (
            f"{contract_part} ; {expanded_question}"
            if self.config.legal_context.enable_legal_query_expansion
            else q_clean
        )

        n_chunks = len(self.index.chunks)
        doc_align_map = self._compute_doc_alignment_scores(contract_part)
        chunk_doc_scores = np.array(
            [doc_align_map[ch.doc_id] for ch in self.index.chunks], dtype=np.float64
        )
        clause_scores = self._compute_clause_scores(question_part, expanded_question)

        lexical_scores = np.zeros(n_chunks, dtype=np.float64)
        dense_scores = np.zeros(n_chunks, dtype=np.float64)

        strat = self.config.retrieval.strategy

        # 1. Compute Lexical Scores (TF-IDF or BM25)
        if strat == "tfidf" or (strat == "hybrid" and self.config.retrieval.hybrid_lexical_type == "tfidf"):
            q_vec = self.index.tfidf_vectorizer.transform([effective_query])
            lexical_scores = (self.index.tfidf_matrix @ q_vec.T).toarray().ravel()
            if use_routed_clause_query:
                lexical_scores = 0.65 * clause_scores + 0.35 * lexical_scores
        elif strat == "bm25" or (strat == "hybrid" and self.config.retrieval.hybrid_lexical_type == "bm25"):
            assert self.index.bm25_index is not None
            q_tokens = legal_tokenize(effective_query, remove_stopwords=True)
            bm25_raw = np.array(self.index.bm25_index.get_scores(q_tokens), dtype=np.float64)
            lexical_scores = bm25_saturation_normalize(bm25_raw, k_sat=8.0)
            if use_routed_clause_query:
                lexical_scores = 0.65 * clause_scores + 0.35 * lexical_scores

        # 2. Compute Dense Scores (SentenceTransformer + FAISS)
        if strat in ("dense", "hybrid"):
            assert self.index.chunk_embeddings is not None
            if use_routed_clause_query:
                q_embs = encode_query_texts(
                    self.config.retrieval.dense_model_name,
                    [expanded_question, q_clean],
                    cache_dir=self.config.cache_dir,
                )
                q_emb = (0.75 * q_embs[0:1] + 0.25 * q_embs[1:2]).astype(np.float32)
                norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
                q_emb = q_emb / np.maximum(norm, 1e-12)
            else:
                q_emb = encode_query_texts(
                    self.config.retrieval.dense_model_name,
                    [q_clean],
                    cache_dir=self.config.cache_dir,
                )
            raw_sims = (self.index.chunk_embeddings @ q_emb.T).ravel()
            dense_scores = np.clip(raw_sims, 0.0, 1.0)

        # 3. Combine scores based on strategy
        if strat == "tfidf":
            combined_scores = lexical_scores.copy()
        elif strat == "bm25":
            combined_scores = lexical_scores.copy()
        elif strat == "dense":
            combined_scores = dense_scores.copy()
        elif strat == "hybrid":
            if self.config.retrieval.fusion_method == "rrf":
                combined_scores = reciprocal_rank_fusion(
                    lexical_scores=lexical_scores,
                    dense_scores=dense_scores,
                    lexical_weight=self.config.retrieval.lexical_weight,
                    dense_weight=self.config.retrieval.dense_weight,
                    rrf_k=self.config.retrieval.rrf_k,
                )
            else:
                combined_scores = linear_score_fusion(
                    lexical_scores=lexical_scores,
                    dense_scores=dense_scores,
                    lexical_type=self.config.retrieval.hybrid_lexical_type,
                    lexical_weight=self.config.retrieval.lexical_weight,
                    dense_weight=self.config.retrieval.dense_weight,
                )
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strat}")

        # 4. Hierarchical document routing + clause-focused score modulation
        w_doc = self.config.legal_context.doc_routing_weight
        if w_doc > 0.0:
            max_doc = float(np.max(chunk_doc_scores)) if chunk_doc_scores.size > 0 else 0.0
            doc_mask = np.where(chunk_doc_scores >= max(0.25, max_doc - 0.08), 1.0, 0.15)
            blend = min(0.35, 0.25 * w_doc)
            combined_scores = doc_mask * (
                (1.0 - blend) * combined_scores + blend * chunk_doc_scores
            )

        pool_size = min(n_chunks, max(k, self.config.retrieval.candidate_pool_size))
        candidate_indices = [int(i) for i in np.argsort(-combined_scores)[:pool_size]]

        # 5. Optional second-stage Cross-Encoder reranking
        rerank_map: dict[int, float] = {}
        if self.config.rerank.enabled:
            rerank_query = expanded_question if ";" in q_clean else q_clean
            candidate_indices, rerank_map = apply_cross_encoder_rerank(
                query_for_rerank=rerank_query,
                chunks=self.index.chunks,
                candidate_indices=candidate_indices,
                base_scores=combined_scores,
                rerank_cfg=self.config.rerank,
            )

        # Build top-k candidate EvidencePassages and verify original character offsets
        top_indices = candidate_indices[:k]
        unfiltered_passages: list[EvidencePassage] = []
        for rank_idx, chunk_idx in enumerate(top_indices, start=1):
            ch = self.index.chunks[chunk_idx]
            if not verify_evidence_offset(
                self.contracts, ch.doc_id, ch.char_start, ch.char_end, ch.raw_text
            ):
                raise AssertionError(
                    f"Evidence offset validation failed for {ch.doc_id}[{ch.char_start}:{ch.char_end}]"
                )
            score_breakdown = {
                "retrieval_score": float(combined_scores[chunk_idx]),
                "lexical_score": float(lexical_scores[chunk_idx]),
                "dense_score": float(dense_scores[chunk_idx]),
                "clause_score": float(clause_scores[chunk_idx]),
                "doc_match_score": float(chunk_doc_scores[chunk_idx]),
            }
            if chunk_idx in rerank_map:
                score_breakdown["rerank_score"] = float(rerank_map[chunk_idx])

            unfiltered_passages.append(
                EvidencePassage(
                    doc_id=ch.doc_id,
                    file_path=ch.file_path,
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                    passage_text=ch.raw_text,
                    retrieval_method=method_label,
                    rank=rank_idx,
                    retrieval_score=float(combined_scores[chunk_idx]),
                    scores=score_breakdown,
                    section_heading=ch.section_heading,
                    clause_number=ch.clause_number,
                    cross_references=ch.cross_references,
                    referenced_context=ch.referenced_context,
                )
            )

        # 6. Evaluate abstention using top candidates
        if unfiltered_passages:
            top_score = unfiltered_passages[0].retrieval_score
            top_doc_id = unfiltered_passages[0].doc_id
            top_doc_match = unfiltered_passages[0].scores["doc_match_score"]
            raw_top_clause = max(
                p.scores["clause_score"]
                for p in unfiltered_passages[: min(3, len(unfiltered_passages))]
                if p.doc_id == top_doc_id
            )
            concept_gate = self._compute_concept_gate_score(
                question_part, unfiltered_passages, top_doc_id
            )
            top_clause = 0.45 * raw_top_clause + 0.55 * concept_gate
        else:
            top_score = 0.0
            top_clause = 0.0
            top_doc_match = 0.0

        decision = evaluate_abstention(
            query=q_clean,
            top_score=top_score,
            top_clause_score=top_clause,
            top_doc_match_score=top_doc_match,
            cfg=self.config.abstention,
        )

        final_passages = [] if decision.abstained else unfiltered_passages
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResponse(
            query=query,
            abstained=decision.abstained,
            abstention=decision,
            passages=final_passages,
            unfiltered_top_passages=unfiltered_passages,
            retrieval_method=method_label,
            latency_ms=elapsed_ms,
            top_score=top_score,
            top_clause_score=top_clause,
            top_doc_match_score=top_doc_match,
        )
