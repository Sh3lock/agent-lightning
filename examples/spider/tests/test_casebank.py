import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_module.casebank import CaseBank
from memory_module.embedder import HashingEmbedder
from memory_module.skeletonizer import SqlSkeletonizer
from memory_module.vector_store import InMemoryVectorStore


def _store_factory(name: str, embedder: HashingEmbedder) -> InMemoryVectorStore:
    return InMemoryVectorStore(name, embedder)


def test_casebank_policy_tiered_vs_skeleton() -> None:
    embedder = HashingEmbedder(dim=64)
    casebank = CaseBank(
        embedder=embedder,
        vector_store_factory=_store_factory,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
    )

    skeletonizer = SqlSkeletonizer()
    sql = "SELECT COUNT(*) FROM users"
    skeleton = skeletonizer.skeletonize(sql, dialect="sqlite").skeleton_sql

    question = "how many users"
    casebank.add_specific(f"Q: {question}\nSQL: {sql}", {"db_id": "db1"})
    casebank.add_skeleton(f"Q: {question}\nSQL: {skeleton}", {"db_id": "global"})

    res = casebank.retrieve_tiered(
        question,
        db_id="db1",
        dialect="sqlite",
        policy="skeleton_only",
        k=1,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
    )
    assert res.type != "specific"

    res_tiered = casebank.retrieve_tiered(
        question,
        db_id="db1",
        dialect="sqlite",
        policy="tiered",
        k=1,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
    )
    assert res_tiered.type == "specific"
