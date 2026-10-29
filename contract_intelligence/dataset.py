"""Dataset loading, exact LegalBench-RAG-mini selection, and prescribed SHA-256 split assignment."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


def mini(tests: list[dict[str, Any]], n: int = 194) -> list[dict[str, Any]]:
    """Reproduce the LegalBench-RAG authors' exact mini benchmark selection (n=194)."""
    if len(tests) <= n:
        return tests

    def key(t: dict[str, Any]) -> float:
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    """Prescribed deterministic SHA-256 train/test split function."""
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


@dataclass(frozen=True)
class GroundTruthSnippet:
    """A single expert-annotated character span in a source contract."""

    file_path: str
    char_start: int
    char_end: int
    answer: str

    @property
    def span(self) -> tuple[int, int]:
        return (self.char_start, self.char_end)


@dataclass(frozen=True)
class QueryRecord:
    """A benchmark or unanswerable query record with split and ground-truth metadata."""

    query_id: str
    query: str
    split: Literal["train", "test"]
    is_answerable: bool
    snippets: list[GroundTruthSnippet] = field(default_factory=list)
    category: str = "cuad_answerable"
    unanswerable_reason: str | None = None


@dataclass(frozen=True)
class ContractDocument:
    """An indexed contract read in Python's default text mode with UTF-8 encoding."""

    doc_id: str
    file_path: str
    text: str
    char_length: int
    title_header: str


@dataclass
class LegalBenchMiniDataset:
    """Container holding the 29 indexed contracts, 194 answerable queries, and 36 unanswerable queries."""

    contracts: dict[str, ContractDocument]
    answerable_queries: list[QueryRecord]
    unanswerable_queries: list[QueryRecord]

    def get_queries(
        self,
        split: Literal["train", "test", "all"] = "train",
        include_answerable: bool = True,
        include_unanswerable: bool = True,
    ) -> list[QueryRecord]:
        """Return queries filtered by split and answerability."""
        combined: list[QueryRecord] = []
        if include_answerable:
            combined.extend(self.answerable_queries)
        if include_unanswerable:
            combined.extend(self.unanswerable_queries)
        if split == "all":
            return combined
        return [q for q in combined if q.split == split]

    def summary_counts(self) -> dict[str, int]:
        """Return exact counts of contracts and train/test answerable and unanswerable queries."""
        ans_train = sum(1 for q in self.answerable_queries if q.split == "train")
        ans_test = sum(1 for q in self.answerable_queries if q.split == "test")
        unans_train = sum(1 for q in self.unanswerable_queries if q.split == "train")
        unans_test = sum(1 for q in self.unanswerable_queries if q.split == "test")
        return {
            "indexed_contracts": len(self.contracts),
            "answerable_total": len(self.answerable_queries),
            "answerable_train": ans_train,
            "answerable_test": ans_test,
            "unanswerable_total": len(self.unanswerable_queries),
            "unanswerable_train": unans_train,
            "unanswerable_test": unans_test,
            "total_queries": len(self.answerable_queries) + len(self.unanswerable_queries),
            "total_train": ans_train + unans_train,
            "total_test": ans_test + unans_test,
        }


def extract_contract_header(doc_id: str, text: str, max_chars: int = 900) -> str:
    """Extract a compact, deterministic header from the contract filename and opening preamble."""
    stem = Path(doc_id).stem.replace("_", " ").replace("-", " ")
    preamble_raw = text[:max_chars]
    lines = [ln.strip() for ln in preamble_raw.splitlines() if ln.strip()]
    preamble_clean = " ".join(lines)[:max_chars]
    return f"[Document: {stem}] [Preamble: {preamble_clean}]"


def read_contract_utf8(path: Path) -> str:
    """Read a contract file using UTF-8 and Python's default text mode (universal newlines)."""
    with open(path, mode="r", encoding="utf-8") as f:
        return f.read()


def resolve_benchmark_path(data_dir: str | Path) -> Path:
    """Locate benchmarks/cuad.json under data_dir or workspace root."""
    base = Path(data_dir)
    candidates = [
        base / "benchmarks" / "cuad.json",
        Path("benchmarks") / "cuad.json",
        Path("data/legalbench_rag/benchmarks/cuad.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find benchmarks/cuad.json in {[str(c) for c in candidates]}. "
        "Run `python -m contract_intelligence download-data` first."
    )


def resolve_corpus_file(data_dir: str | Path, rel_file_path: str) -> Path:
    """Locate a corpus contract file under data_dir/corpus or workspace corpus/."""
    base = Path(data_dir)
    candidates = [
        base / "corpus" / rel_file_path,
        Path("corpus") / rel_file_path,
        Path("data/legalbench_rag/corpus") / rel_file_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Contract file '{rel_file_path}' not found in {[str(c) for c in candidates]}."
    )


def download_legalbench_cuad(data_dir: str | Path = "data/legalbench_rag") -> dict[str, Any]:
    """Download benchmarks/cuad.json and the 29 referenced CUAD-mini contract files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_file = out_dir / "benchmarks" / "cuad.json"
    if not bench_file.exists():
        hf_hub_download(
            repo_id="awinml/legalbench-rag",
            filename="benchmarks/cuad.json",
            repo_type="dataset",
            local_dir=str(out_dir),
        )

    raw = read_contract_utf8(bench_file)
    tests = json.loads(raw)["tests"]
    cuad_mini = mini(tests, n=194)
    referenced_files = sorted({s["file_path"] for t in cuad_mini for s in t["snippets"]})

    downloaded = 0
    for rel_path in referenced_files:
        target = out_dir / "corpus" / rel_path
        if not target.exists():
            hf_hub_download(
                repo_id="awinml/legalbench-rag",
                filename=f"corpus/{rel_path}",
                repo_type="dataset",
                local_dir=str(out_dir),
            )
            downloaded += 1

    return {
        "benchmark_file": str(bench_file),
        "total_cuad_queries": len(tests),
        "mini_queries": len(cuad_mini),
        "referenced_contracts": len(referenced_files),
        "newly_downloaded_contracts": downloaded,
    }


def load_unanswerable_queries(
    unanswerable_path: str | Path = "data/unanswerable_queries.json",
) -> list[QueryRecord]:
    """Load the committed unanswerable query set and verify split assignments."""
    path = Path(unanswerable_path)
    if not path.exists():
        raise FileNotFoundError(f"Unanswerable query file not found: {path}")
    payload = json.loads(read_contract_utf8(path))
    raw_list = payload.get("queries", [])
    records: list[QueryRecord] = []
    for item in raw_list:
        qtext = item["query"]
        computed_split = assign_split(qtext)
        stored_split = item.get("split", computed_split)
        if stored_split != computed_split:
            raise ValueError(
                f"Split mismatch for unanswerable query {item['query_id']}: "
                f"stored={stored_split} vs assign_split={computed_split}"
            )
        records.append(
            QueryRecord(
                query_id=item["query_id"],
                query=qtext,
                split=computed_split,  # type: ignore[arg-type]
                is_answerable=False,
                snippets=[],
                category=item.get("category", "unanswerable"),
                unanswerable_reason=item.get("reason", "Unsupported by indexed corpus."),
            )
        )
    return records


def load_dataset(
    data_dir: str | Path = "data/legalbench_rag",
    unanswerable_path: str | Path = "data/unanswerable_queries.json",
    header_char_limit: int = 900,
) -> LegalBenchMiniDataset:
    """Load the exact 194 LegalBench-RAG-mini CUAD queries, ONLY their 29 referenced contracts, and unanswerable queries."""
    bench_path = resolve_benchmark_path(data_dir)
    raw = read_contract_utf8(bench_path)
    tests = json.loads(raw)["tests"]
    cuad_mini_raw = mini(tests, n=194)

    answerable_queries: list[QueryRecord] = []
    referenced_file_paths: set[str] = set()

    for idx, item in enumerate(cuad_mini_raw):
        qtext = item["query"]
        qsplit = assign_split(qtext)
        snippets: list[GroundTruthSnippet] = []
        for s in item["snippets"]:
            fp = s["file_path"]
            referenced_file_paths.add(fp)
            start, end = int(s["span"][0]), int(s["span"][1])
            snippets.append(
                GroundTruthSnippet(
                    file_path=fp,
                    char_start=start,
                    char_end=end,
                    answer=s.get("answer", ""),
                )
            )
        # Extract clause category from the question part after ';'
        q_part = qtext.split(";", 1)[1].strip() if ";" in qtext else qtext
        answerable_queries.append(
            QueryRecord(
                query_id=f"CUAD-MINI-{idx + 1:03d}",
                query=qtext,
                split=qsplit,  # type: ignore[arg-type]
                is_answerable=True,
                snippets=snippets,
                category=q_part,
            )
        )

    # Index ONLY the contracts referenced by the 194 mini queries
    contracts: dict[str, ContractDocument] = {}
    for rel_fp in sorted(referenced_file_paths):
        full_path = resolve_corpus_file(data_dir, rel_fp)
        content = read_contract_utf8(full_path)
        contracts[rel_fp] = ContractDocument(
            doc_id=rel_fp,
            file_path=str(full_path),
            text=content,
            char_length=len(content),
            title_header=extract_contract_header(rel_fp, content, max_chars=header_char_limit),
        )

    unanswerable_queries = load_unanswerable_queries(unanswerable_path)

    return LegalBenchMiniDataset(
        contracts=contracts,
        answerable_queries=answerable_queries,
        unanswerable_queries=unanswerable_queries,
    )

