#!/usr/bin/env python3
"""Extract simple capacitance model from a Liberty file.

This script is intentionally standalone and compatible with Python <3.9.
It parses a Liberty ``.lib`` (or gzip-compressed ``.lib.gz``) file and writes a
JSON mapping each cell name to its total input pin capacitance.

Conveniences:
- Dump all per-cell totals (default behavior).
- Query a single cell name or generic cell type (e.g., NOT, NAND).
- Dump a generic mapping like the default Yosys "notech" model (BUF, NOT, AND, ...).
"""

import json
import re
import sys
import gzip
import argparse


def parse_liberty(path):
    cell_re = re.compile(r"cell\s*\(([^)]+)\)")
    pin_re = re.compile(r"pin\s*\(([^)]+)\)")
    cap_re = re.compile(r"capacitance\s*:\s*([0-9eE.+-]+)")
    dir_re = re.compile(r"direction\s*:\s*(\w+)")

    result = {}
    cell = None
    pin = None
    in_pin = False

    def _open_text_auto(p):
        # Detect gzip by magic number to support both .lib and .lib.gz
        try:
            with open(p, "rb") as fb:
                magic = fb.read(2)
        except OSError:
            # Let the subsequent open raise a clearer error
            magic = b""
        if magic == b"\x1f\x8b" or p.endswith(".gz"):
            return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
        return open(p, "rt", encoding="utf-8", errors="ignore")

    with _open_text_auto(path) as f:
        for line in f:
            m = cell_re.search(line)
            if m:
                cell = m.group(1)
                result[cell] = {}
                continue
            m = pin_re.search(line)
            if m:
                pin = m.group(1)
                in_pin = True
                continue
            if in_pin:
                m = dir_re.search(line)
                if m:
                    in_pin = m.group(1) in ("input", "inout")
                m = cap_re.search(line)
                if m and in_pin and cell is not None and pin is not None:
                    result[cell][pin] = float(m.group(1))
                    pin = None
                    in_pin = False
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-cell input pin capacitance totals from Liberty (.lib or .lib.gz).\n"
            "Default mode writes a JSON mapping of all cells to totals."
        )
    )
    # Backward-compatible positional arguments for default (dump_all) mode
    parser.add_argument("input", nargs="?", help="Path to Liberty .lib or .lib.gz")
    parser.add_argument("output", nargs="?", help="Output JSON path (for default dump)")

    mode_group = parser.add_argument_group("modes", "Optional query modes")
    mode_group.add_argument("--cell-name", dest="cell_name", help="Exact cell name to query")
    mode_group.add_argument(
        "--cell-type",
        dest="cell_type",
        help=(
            "Generic type to query representative smallest-drive cell (e.g., BUF, NOT, AND, NAND, OR, NOR, XOR, XNOR, MUX, NMUX, AOI3, OAI3, AOI4, OAI4, DFF_PN0)"
        ),
    )
    mode_group.add_argument(
        "--list-type",
        dest="list_type",
        help="List all cells matching a generic type (writes JSON map)",
    )
    mode_group.add_argument(
        "--dump-generic",
        dest="dump_generic",
        action="store_true",
        help="Dump generic mapping across common types (BUF/NOT/AND/...)",
    )

    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        help="Optional output JSON path (stdout if omitted for query modes)",
    )

    args = parser.parse_args()

    # Determine mode
    mode = "dump_all"
    if args.cell_name:
        mode = "cell_name"
    if args.cell_type:
        mode = "cell_type"
    if args.list_type:
        mode = "list_type"
    if args.dump_generic:
        mode = "dump_generic"

    # Validate arguments per mode and set paths
    if mode == "dump_all":
        if not args.input or not args.output:
            parser.error("default mode requires <input> and <output>")
        in_path = args.input
        out_path = args.output
    else:
        if not args.input:
            parser.error("query modes require <input>")
        in_path = args.input
        out_path = args.out or args.output  # allow either -o or trailing positional

    data = parse_liberty(in_path)
    totals = {cell: sum(pins.values()) for cell, pins in data.items()}

    # Default: dump all per-cell totals to JSON
    if mode == "dump_all":
        with open(out_path, "w") as fp:
            json.dump(totals, fp, indent=2, sort_keys=True)
        return

    # Helpers for generic cell type matching and representative selection
    def parse_drive_strength(name):
        n = name
        u = n.upper()
        if "X" not in u:
            return None
        # Find last occurrence to skip AOI22-like tokens
        x_idx = u.rfind("X")
        if x_idx == -1 or x_idx + 1 >= len(u):
            return None
        i2 = x_idx + 1
        digits = []
        saw_digit = False
        saw_p = False
        while i2 < len(u):
            ch = u[i2]
            if ch.isdigit():
                digits.append(ch)
                saw_digit = True
                i2 += 1
                continue
            if ch == "P":
                # decimal point marker used in some libs (e.g., x1p5 => 1.5, xp33 => 0.33)
                digits.append(".")
                saw_p = True
                i2 += 1
                continue
            break
        if not digits:
            return None
        s = "".join(digits)
        try:
            val = float(s)
            # Special-case: patterns like xp33 mean 0.33 (no leading digits)
            if saw_p and not saw_digit:
                val = val / (10 ** (len(s.split(".")[-1])))
            return val
        except Exception:
            return None

    SYNONYMS = {
        "BUF": ["BUF"],
        "NOT": ["INV", "CKINV", "NOT"],
        "AND": ["AND"],
        "NAND": ["NAND"],
        "OR": ["OR"],
        "NOR": ["NOR"],
        "XOR": ["XOR"],
        "XNOR": ["XNOR"],
        "MUX": ["MUX"],
        "NMUX": ["NMUX"],
        "AOI3": ["AOI3", "AOI21", "AOI31"],
        "OAI3": ["OAI3", "OAI21", "OAI31"],
        "AOI4": ["AOI4", "AOI22", "AOI211"],
        "OAI4": ["OAI4", "OAI22", "OAI211"],
        "DFF_PN0": ["DFF"],
        # "ANDNOT"/"ORNOT" rarely appear verbatim in cells; omit dedicated synonyms
        "ANDNOT": ["ANDNOT"],
        "ORNOT": ["ORNOT"],
    }

    def cell_matches_type(cell, typ):
        u = cell.upper()
        typu = typ.upper()
        # Avoid false positives for OR / AND against XOR/NOR/NAND
        if typu == "OR":
            return ("OR" in u) and ("XOR" not in u) and ("NOR" not in u)
        if typu == "AND":
            return ("AND" in u) and ("NAND" not in u)
        syns = SYNONYMS.get(typu, [typu])
        return any(s in u for s in syns)

    def representative_for_type(typ):
        # Return (cell, cap) choosing smallest drive when available, else min cap
        matches = [(c, cap) for c, cap in totals.items() if cell_matches_type(c, typ)]
        if not matches:
            return None, None
        # Try by minimum parsed drive
        with_drive = []
        without_drive = []
        for c, cap in matches:
            d = parse_drive_strength(c)
            if d is None:
                without_drive.append((c, cap))
            else:
                with_drive.append((d, c, cap))
        if with_drive:
            with_drive.sort(key=lambda t: (t[0], t[2]))  # by drive, then cap
            _d, c, cap = with_drive[0]
            return c, cap
        # Fallback: minimum capacitance
        c, cap = min(without_drive, key=lambda t: t[1])
        return c, cap

    if mode == "cell_name":
        cap = totals.get(args.cell_name)
        if cap is None:
            print(f"Cell not found: {args.cell_name}", file=sys.stderr)
            sys.exit(2)
        obj = {"cell": args.cell_name, "capacitance": cap}
        if out_path:
            with open(out_path, "w") as fp:
                json.dump(obj, fp, indent=2, sort_keys=True)
        else:
            print(json.dumps(obj, indent=2, sort_keys=True))
        return

    if mode == "cell_type":
        rep_cell, cap = representative_for_type(args.cell_type)
        if rep_cell is None:
            print(f"No match for type: {args.cell_type}", file=sys.stderr)
            sys.exit(2)
        obj = {"type": args.cell_type, "cell": rep_cell, "capacitance": cap}
        if out_path:
            with open(out_path, "w") as fp:
                json.dump(obj, fp, indent=2, sort_keys=True)
        else:
            print(json.dumps(obj, indent=2, sort_keys=True))
        return

    if mode == "list_type":
        filtered = {c: cap for c, cap in totals.items() if cell_matches_type(c, args.list_type)}
        if not filtered:
            print(f"No match for type: {args.list_type}", file=sys.stderr)
            sys.exit(2)
        if out_path:
            with open(out_path, "w") as fp:
                json.dump(filtered, fp, indent=2, sort_keys=True)
        else:
            print(json.dumps(filtered, indent=2, sort_keys=True))
        return

    if mode == "dump_generic":
        if not out_path:
            print("--dump-generic requires an output path (-o/--out or positional)", file=sys.stderr)
            sys.exit(2)
        generic_types = [
            "BUF",
            "NOT",
            "AND",
            "NAND",
            "OR",
            "NOR",
            "ANDNOT",
            "ORNOT",
            "XOR",
            "XNOR",
            "AOI3",
            "OAI3",
            "AOI4",
            "OAI4",
            "MUX",
            "NMUX",
            "DFF_PN0",
        ]
        out = {}
        for t in generic_types:
            c, cap = representative_for_type(t)
            if c is not None and cap is not None:
                out[t] = cap
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2, sort_keys=True)
        return


if __name__ == "__main__":
    main()
