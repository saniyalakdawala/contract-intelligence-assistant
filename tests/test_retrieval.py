"""Automated tests for retrieval strategies, empty/malformed queries, config validation, and ranking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from contract_intelligence.config import ChunkingConfig, PipelineConfig, RetrievalConfig
from contract_intelligence.dataset import ContractDocument
from contract_intelligence.retrieval import ClauseTraceRetriever
from contract_intelligence.validation import verify_evidence_offset


def _synthetic_contracts() -> dict[str, ContractDocument]:
    txt1 = (
        "SOFTWARE LICENSE AGREEMENT between AlphaCorp Inc. and BetaSoft LLC.\n\n"
        "SECTION 1. SUBLICENSING\n"
        "1.1 Licensee may not sublicense, assign, or transfer the Software to any third-party platform "
        "without the prior written consent of AlphaCorp Inc.\n\n"
        "SECTION 2. GOVERNING LAW\n"
        "2.1 This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.\n"
    )
    txt2 = (
        "SPONSORSHIP AGREEMENT between GammaSports Ltd. and DeltaBeverage Corp.\n\n"
        "SECTION 1. TERM AND EXPIRATION\n"
        "1.1 The exclusivity period and sponsorship term shall expire on December 31, 2027.\n\n"
        "SECTION 2. INTELLECTUAL PROPERTY\n"
        "2.1 GammaSports Ltd. shall own all intellectual property rights in any derivative works created hereunder.\n"
    )
    return {
        "synthetic/alphacorp_betasoft.txt": ContractDocument(
            doc_id="synthetic/alphacorp_betasoft.txt",
            file_path="synthetic/alphacorp_betasoft.txt",
            text=txt1,
            char_length=len(txt1),
            title_header="[Document: alphacorp betasoft] [Preamble: SOFTWARE LICENSE AGREEMENT between AlphaCorp Inc. and BetaSoft LLC.]",
        ),
        "synthetic/gammasports_deltabeverage.txt": ContractDocument(
            doc_id="synthetic/gammasports_deltabeverage.txt",
            file_path="synthetic/gammasports_deltabeverage.txt",
            text=txt2,
            char_length=len(txt2),
            title_header="[Document: gammasports deltabeverage] [Preamble: SPONSORSHIP AGREEMENT between GammaSports Ltd. and DeltaBeverage Corp.]",
        ),
    }


def test_invalid_configurations_rejected() -> None:
    """Verify invalid chunking and retrieval configurations raise ValidationError."""
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=300)

    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=300, chunk_overlap=100, min_chunk_size=400)

    with pytest.raises(ValidationError):
        RetrievalConfig(tfidf_ngram_min=2, tfidf_ngram_max=1)

    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=20, candidate_pool_size=10)


def test_empty_and_malformed_queries_abstain_cleanly() -> None:
    """Verify empty, whitespace, or punctuation-only queries abstain without raising errors."""
    contracts = _synthetic_contracts()
    cfg = PipelineConfig()
    retriever = ClauseTraceRetriever(contracts, cfg, use_cache=False)

    for bad_q in ["", "   ", "???", "@#$%^&*"]:
        resp = retriever.search(bad_q, top_k=3)
        assert resp.abstained is True
        assert resp.abstention.status == "abstained_invalid_query"
        assert len(resp.passages) == 0


def test_tfidf_and_bm25_retrieval_ranking_and_offset_integrity() -> None:
    """Verify TF-IDF and BM25 rank relevant clauses first and preserve exact original character offsets."""
    contracts = _synthetic_contracts()
    for strat in ("tfidf", "bm25"):
        cfg = PipelineConfig(
            chunking=ChunkingConfig(strategy="clause_aware", chunk_size=220, chunk_overlap=30, min_chunk_size=30),
            retrieval=RetrievalConfig(strategy=strat, top_k=3, candidate_pool_size=5),  # type: ignore[arg-type]
        )
        cfg.legal_context.include_doc_header = True
        cfg.legal_context.doc_routing_weight = 0.85
        cfg.abstention.min_top_score = 0.05

        retriever = ClauseTraceRetriever(contracts, cfg, use_cache=False)
        q = "Consider the Software License Agreement between AlphaCorp Inc. and BetaSoft LLC; What is the governing law for this contract?"
        resp = retriever.search(q, top_k=2)

        assert not resp.abstained
        assert len(resp.passages) >= 1
        top = resp.passages[0]
        assert top.doc_id == "synthetic/alphacorp_betasoft.txt"
        assert "Delaware" in top.passage_text
        assert verify_evidence_offset(
            contracts, top.doc_id, top.char_start, top.char_end, top.passage_text
        )

        # Verify ranking consistency across repeated queries
        resp2 = retriever.search(q, top_k=2)
        assert [p.passage_text for p in resp.passages] == [p.passage_text for p in resp2.passages]

