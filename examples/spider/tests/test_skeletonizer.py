import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_module.skeletonizer import SqlSkeletonizer


def test_basic_skeleton_no_leak() -> None:
    skeletonizer = SqlSkeletonizer()
    result = skeletonizer.skeletonize(
        "SELECT name FROM users WHERE age > 18",
        dialect="sqlite",
    )

    assert result.failed is False
    skeletonizer.assert_no_identifiers(
        "SELECT name FROM users WHERE age > 18",
        result.skeleton_sql,
        dialect="sqlite",
    )
    lowered = result.skeleton_sql.lower()
    for token in ("users", "name", "age", "18"):
        assert token not in lowered
    for keyword in ("select", "from", "where"):
        assert keyword in lowered


def test_column_mapping_consistent() -> None:
    skeletonizer = SqlSkeletonizer()
    sql = "SELECT u.id, o.user_id FROM users u JOIN orders o ON u.id = o.user_id"
    result = skeletonizer.skeletonize(sql, dialect="sqlite")

    assert result.failed is False
    cols = re.findall(r"_col\\d+", result.skeleton_sql)
    assert cols
    unique = {c: cols.count(c) for c in cols}
    assert len(unique) == 2
    assert all(count >= 2 for count in unique.values())
