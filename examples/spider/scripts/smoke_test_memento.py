import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_module.casebank import CaseBank
from memory_module.embedder import HashingEmbedder
from memory_module.error_fix_bank import ErrorFixBank
from memory_module.error_normalizer import normalize_error
from memory_module.skeletonizer import SqlSkeletonizer
from memory_module.validator import StaticValidator
from memory_module.vector_store import InMemoryVectorStore
from sql_agent import _build_table_info_with_memory_context, _append_static_validation_feedback


def _store_factory(name: str, embedder: HashingEmbedder) -> InMemoryVectorStore:
    return InMemoryVectorStore(name, embedder)


def main() -> None:
    skeletonizer = SqlSkeletonizer()
    sql = "SELECT name FROM users WHERE age > 18"
    result = skeletonizer.skeletonize(sql, dialect="sqlite")
    if result.failed:
        raise SystemExit(f"skeletonizer failed: {result.op_signature}")
    skeletonizer.assert_no_identifiers(sql, result.skeleton_sql, dialect="sqlite")
    lowered = result.skeleton_sql.lower()
    for token in ("users", "name", "age", "18"):
        assert token not in lowered
    for keyword in ("select", "from", "where"):
        assert keyword in lowered

    embedder = HashingEmbedder(dim=64)
    casebank = CaseBank(
        embedder=embedder,
        vector_store_factory=_store_factory,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
        allow_splits=[],
    )
    question = "how many users"
    specific_sql = "SELECT COUNT(*) FROM users"
    skeleton_sql = skeletonizer.skeletonize(specific_sql, dialect="sqlite").skeleton_sql
    casebank.add_specific(f"Q: {question}\nSQL: {specific_sql}", {"db_id": "db1"})
    casebank.add_skeleton(f"Q: {question}\nSQL: {skeleton_sql}", {"db_id": "global"})

    res = casebank.retrieve_tiered(
        question=question,
        db_id="db1",
        dialect="sqlite",
        policy="skeleton_only",
        k=1,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
    )
    assert res.type in {"skeleton", "none"}

    res_tiered = casebank.retrieve_tiered(
        question=question,
        db_id="db1",
        dialect="sqlite",
        policy="tiered",
        k=1,
        min_score_specific=0.0,
        min_score_skeleton=0.0,
    )
    assert res_tiered.type == "specific"

    table_info = "table users(id, name, age)"
    memory_context = "### Relevant Past Cases (Same Database)\nExample"
    final_table_info = _build_table_info_with_memory_context(memory_context, table_info, max_chars=64)
    assert "Current Schema" in final_table_info
    assert table_info in final_table_info

    normalized = normalize_error("OperationalError: no such column: flight_id", dialect="sqlite")
    assert normalized.error_type == "MissingColumn"
    assert "flight_id" in normalized.entities.get("columns", [])

    fix_bank = ErrorFixBank(
        embedder=embedder,
        vector_store_factory=_store_factory,
        min_score=0.0,
    )
    fix_bank.add_fix_hint(
        "MissingColumn: verify column names in CURRENT SCHEMA and add table prefix if needed.",
        {"error_type": "MissingColumn", "dialect": "sqlite"},
    )
    fix_bank.add_fix_hint(
        "MissingColumn: check if the column lives in a joined table.",
        {"error_type": "MissingColumn", "dialect": "sqlite"},
    )
    hints = fix_bank.retrieve_fix_hints(
        error_type="MissingColumn",
        dialect="sqlite",
        query_text="MissingColumn\nOperationalError: no such column: flight_id",
        k=2,
        min_score=0.0,
    )
    assert hints

    validator = StaticValidator()
    table_info = "CREATE TABLE users (id INTEGER, name TEXT);"
    validation = validator.validate("SELECT age FROM users", table_info, "sqlite")
    if validation.error_type != "Unavailable":
        assert validation.ok is False
        assert validation.error_type == "MissingColumn"

    raw_feedback = "Looks good.\nTHE QUERY IS CORRECT."
    forced = _append_static_validation_feedback(raw_feedback, "Missing column: age")
    assert "THE QUERY IS CORRECT." not in forced
    assert forced.rstrip().endswith("THE QUERY IS INCORRECT.")
    print("Memento smoke test passed.")


if __name__ == "__main__":
    main()
