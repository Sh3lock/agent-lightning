from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import re

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SkeletonResult:
    skeleton_sql: str
    op_signature: str
    entities: Dict[str, Dict[str, str]]
    failed: bool


class SqlSkeletonizer:
    def __init__(self, default_dialect: str = "sqlite") -> None:
        self.default_dialect = default_dialect

    def skeletonize(self, sql: str, dialect: Optional[str] = None) -> SkeletonResult:
        if sqlglot is None or exp is None:
            return SkeletonResult(
                skeleton_sql="",
                op_signature="MISSING_SQLGLOT",
                entities={"tables": {}, "columns": {}, "literals": {}},
                failed=True,
            )
        read_dialect = dialect or self.default_dialect
        write_dialect = dialect or self.default_dialect
        entities: Dict[str, Dict[str, str]] = {"tables": {}, "columns": {}, "literals": {}}
        try:
            tree = sqlglot.parse_one(sql, read=read_dialect)
        except Exception:
            return SkeletonResult(skeleton_sql="", op_signature="PARSE_ERROR", entities=entities, failed=True)

        table_map: Dict[str, str] = {}
        table_alias_map: Dict[str, str] = {}
        col_map: Dict[str, str] = {}
        col_alias_map: Dict[str, str] = {}
        literal_map = {
            "num": "_val_num",
            "str": "_val_str",
            "date": "_val_date",
            "bool": "_val_bool",
        }

        counters = {"tab": 0, "col": 0, "alias": 0}

        def _next(kind: str, prefix: str) -> str:
            counters[kind] += 1
            return f"{prefix}{counters[kind]}"

        def _table_placeholder(name: str) -> str:
            if name not in table_map:
                table_map[name] = _next("tab", "_tab")
            return table_map[name]

        def _col_placeholder(key: str) -> str:
            if key not in col_map:
                col_map[key] = _next("col", "_col")
            return col_map[key]

        def _alias_placeholder(name: str) -> str:
            if name not in col_alias_map:
                col_alias_map[name] = _next("alias", "_alias")
            return col_alias_map[name]

        def _literal_placeholder(node: exp.Literal) -> exp.Expression:
            # Keep placeholders as string literals to preserve SQL validity across contexts.
            if node.is_string:
                entities["literals"]["str"] = literal_map["str"]
                return exp.Literal.string(literal_map["str"])
            if node.is_number:
                entities["literals"]["num"] = literal_map["num"]
                return exp.Literal.string(literal_map["num"])
            entities["literals"]["other"] = "_val_other"
            return exp.Literal.string("_val_other")

        def _replace(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Table):
                name = node.name
                placeholder = _table_placeholder(name)
                alias = node.args.get("alias")
                new_alias = None
                if alias and alias.this:
                    alias_name = alias.this.name
                    table_alias_map[alias_name] = placeholder
                    new_alias = exp.TableAlias(this=exp.Identifier(this=placeholder))
                return exp.Table(this=exp.Identifier(this=placeholder), alias=new_alias)

            if isinstance(node, exp.Subquery):
                alias = node.args.get("alias")
                if alias and alias.this:
                    alias_name = alias.this.name
                    placeholder = _table_placeholder(alias_name)
                    table_alias_map[alias_name] = placeholder
                    new_alias = exp.TableAlias(this=exp.Identifier(this=placeholder))
                    return exp.Subquery(this=node.this, alias=new_alias)
                return node

            if isinstance(node, exp.CTE):
                alias = node.alias
                if alias:
                    placeholder = _table_placeholder(alias)
                    table_alias_map[alias] = placeholder
                    new_alias = exp.TableAlias(this=exp.Identifier(this=placeholder))
                    return exp.CTE(this=node.this, alias=new_alias)
                return node

            if isinstance(node, exp.Column):
                col_name = node.name
                table_name = node.table
                table_placeholder = None
                if table_name:
                    table_placeholder = table_alias_map.get(table_name) or table_map.get(table_name)
                    if table_placeholder is None:
                        table_placeholder = _table_placeholder(table_name)
                key = f"{table_name}.{col_name}" if table_name else col_name
                col_placeholder = _col_placeholder(key)
                entities["columns"][key] = col_placeholder
                return exp.Column(
                    this=exp.Identifier(this=col_placeholder),
                    table=exp.Identifier(this=table_placeholder) if table_placeholder else None,
                )

            if isinstance(node, exp.Alias):
                alias = node.args.get("alias")
                if alias and alias.this:
                    alias_name = alias.this.name
                    alias_placeholder = _alias_placeholder(alias_name)
                    return exp.Alias(this=node.this, alias=exp.Identifier(this=alias_placeholder))
                return node

            if isinstance(node, exp.Literal):
                return _literal_placeholder(node)

            if isinstance(node, exp.Boolean):
                entities["literals"]["bool"] = literal_map["bool"]
                return exp.Literal.string(literal_map["bool"])

            return node

        skeleton_tree = tree.transform(_replace)
        for name, placeholder in table_map.items():
            entities["tables"][name] = placeholder

        op_signature = _build_op_signature(tree)
        skeleton_sql = skeleton_tree.sql(dialect=write_dialect)
        return SkeletonResult(
            skeleton_sql=skeleton_sql,
            op_signature=op_signature,
            entities=entities,
            failed=False,
        )

    def assert_no_identifiers(
        self,
        original_sql: str,
        skeleton_sql: str,
        dialect: Optional[str] = None,
    ) -> None:
        if sqlglot is None or exp is None:
            return
        read_dialect = dialect or self.default_dialect
        try:
            tree = sqlglot.parse_one(original_sql, read=read_dialect)
        except Exception:
            return

        identifiers: set[str] = set()
        for table in tree.find_all(exp.Table):
            if table.name:
                identifiers.add(table.name)
            alias = table.args.get("alias")
            if alias and alias.this:
                identifiers.add(alias.this.name)
        for col in tree.find_all(exp.Column):
            if col.name:
                identifiers.add(col.name)
            if col.table:
                identifiers.add(col.table)
        for ident in tree.find_all(exp.Identifier):
            if ident.name:
                identifiers.add(ident.name)
        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                identifiers.add(cte.alias)
        for subquery in tree.find_all(exp.Subquery):
            alias = subquery.args.get("alias")
            if alias and alias.this:
                identifiers.add(alias.this.name)

        keywords = {
            "select",
            "from",
            "where",
            "join",
            "on",
            "group",
            "by",
            "having",
            "order",
            "limit",
            "union",
            "exists",
            "and",
            "or",
            "as",
            "inner",
            "left",
            "right",
            "full",
            "outer",
            "cross",
            "distinct",
        }
        filtered = {
            name
            for name in identifiers
            if name
            and name.lower() not in keywords
            and not name.startswith("_")
            and re.fullmatch(r"[A-Za-z0-9_]+", name)
        }

        lower_skeleton = skeleton_sql.lower()
        for name in filtered:
            if re.search(rf"\\b{re.escape(name.lower())}\\b", lower_skeleton):
                raise AssertionError(f"identifier leaked into skeleton: {name}")


def _build_op_signature(tree: "exp.Expression") -> str:
    if exp is None:  # pragma: no cover
        return "UNKNOWN"
    ops = [tree.key.upper()]
    checks: list[tuple[str, Any]] = [
        ("JOIN", exp.Join),
        ("GROUPBY", exp.Group),
        ("HAVING", exp.Having),
        ("ORDER", exp.Order),
        ("LIMIT", exp.Limit),
        ("UNION", exp.Union),
        ("EXISTS", exp.Exists),
        ("SUBQUERY", exp.Subquery),
    ]
    window_cls = getattr(exp, "Window", None)
    if window_cls is not None:
        checks.append(("WINDOW", window_cls))
    for label, cls in checks:
        try:
            if tree.find(cls):
                ops.append(label)
        except Exception:
            continue
    return "|".join(dict.fromkeys(ops))
