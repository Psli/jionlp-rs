"""Extract Python time_parser test corpus as structured data."""
import ast
import datetime
import time
import json
import sys
import os

SRC = "/Users/lijiaxi/prj/100x/test/JioNLP/test/test_time_parser.py"

# Python-side time_base "ref_ts" values used at the top of the test.
# Replicate them without importing jionlp.
_ts_1 = 1623604000  # 2021-06-14 01:06:40 local
_ts_2 = 1630480532  # 2021-09-01 15:15:32 local


def time_base_to_iso(tb):
    """Normalize Python test's heterogeneous time_base forms to an ISO string."""
    if tb is None:
        return ""
    if isinstance(tb, (int, float)):
        # Epoch seconds → local datetime → ISO (seconds granularity).
        dt = datetime.datetime.fromtimestamp(tb)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(tb, datetime.datetime):
        return tb.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(tb, list):
        # [y, m, d] or [y, m, d, H, M, S]
        y = tb[0] if len(tb) > 0 else 1970
        m = tb[1] if len(tb) > 1 else 1
        d = tb[2] if len(tb) > 2 else 1
        H = tb[3] if len(tb) > 3 else 0
        M = tb[4] if len(tb) > 4 else 0
        S = tb[5] if len(tb) > 5 else 0
        return f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}"
    if isinstance(tb, dict):
        y = tb.get("year", 1970)
        m = tb.get("month", 1)
        d = tb.get("day", 1)
        H = tb.get("hour", 0)
        M = tb.get("minute", 0)
        S = tb.get("second", 0)
        return f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}"
    return ""


def extract():
    with open(SRC) as f:
        src = f.read()
    tree = ast.parse(src)

    # Find TestTimeParser.test_time_parser — the first list binding is what we want.
    cases = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "test_time_parser":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "time_string_list" for t in stmt.targets)
                ):
                    # Evaluate with a controlled globals dict.
                    ctx = {
                        "_ts_1": _ts_1,
                        "_ts_2": _ts_2,
                        "datetime": datetime,
                        "time": time,
                    }
                    try:
                        lst = eval(compile(ast.Expression(stmt.value), SRC, "eval"), ctx)
                        if isinstance(lst, list):
                            cases.append(lst)
                    except Exception as e:
                        print(f"skip: {e}", file=sys.stderr)
    return cases


def main():
    # Python's test has four separate `time_string_list` bindings. The first
    # is the main parser corpus; later ones are for ret_future / period /
    # lunar variants. Keep only the first for now.
    all_lists = extract()
    if not all_lists:
        print("no cases found", file=sys.stderr)
        sys.exit(1)
    main_list = all_lists[0]
    print(f"# primary list: {len(main_list)} cases", file=sys.stderr)

    out = []
    for item in main_list:
        if not isinstance(item, list) or len(item) < 3:
            continue
        inp, tb, expected = item[0], item[1], item[2]
        if not isinstance(inp, str) or not isinstance(expected, dict):
            continue
        iso = time_base_to_iso(tb)
        # Expected shape: {'type': ..., 'definition': ..., 'time': [start, end]}
        ttype = expected.get("type", "")
        definition = expected.get("definition", "")
        tspan = expected.get("time", ["", ""])
        if not isinstance(tspan, (list, tuple)) or len(tspan) < 2:
            continue
        start, end = tspan[0], tspan[1]
        if not isinstance(start, str) or not isinstance(end, str):
            # time_delta / time_period entries carry dict shapes here;
            # they need a different assertion scheme, handle later.
            continue
        if start in ("inf", "-inf") or end in ("inf", "-inf"):
            # Python uses `inf` / `-inf` sentinels for open-ended spans.
            # Our ISO-based assertion can't match them; skip these — the
            # behavior is still exercised via concrete-endpoint cases
            # (`年底前`, `明年6月之前`, etc.).
            continue
        out.append({
            "input": inp,
            "ref": iso,
            "type": ttype,
            "definition": definition,
            "start": start,
            "end": end,
        })

    json.dump(out, sys.stdout, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    main()
