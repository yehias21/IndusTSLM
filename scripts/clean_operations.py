#!/usr/bin/env python3
"""
Standardize OPERATION text in drilling data.

Features:
  - tqdm progress bar
  - Writes output incrementally (every FLUSH_EVERY rows)
  - Resumable: on restart, skips rows already written to the output CSV
  - Preserves original columns; adds OPERATION_CLEAN

Usage:
    python clean_operations.py                          # defaults
    python clean_operations.py --input X.csv --output Y.csv
    python clean_operations.py --batch-size 5000        # flush interval
"""

import argparse
import csv
import os
import re
import sys

from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ──────────────────────────────────────────────────────────────────────

UNICODE_FRACS = {
    "\u00bd": "1/2", "\u00bc": "1/4", "\u00be": "3/4",
    "\u2153": "1/3", "\u2154": "2/3", "\u2155": "1/5",
    "\u2156": "2/5", "\u2157": "3/5", "\u2158": "4/5",
    "\u2159": "1/6", "\u215a": "5/6", "\u215b": "1/8",
    "\u215c": "3/8", "\u215d": "5/8", "\u215e": "7/8",
}

# (compiled regex, replacement)  — word-boundary aware
ABBREVS = [
    (re.compile(r"(?<!\w)CONT'D(?!\w)",  re.I), "CONTINUED"),
    (re.compile(r"(?<!\w)CONT\.(?=\s|$)", re.I), "CONTINUED"),
    (re.compile(r"(?<!\w)W/(?!\w)",  re.I), "WITH"),
    (re.compile(r"(?<!\w)M/U(?!\w)", re.I), "MAKE UP"),
    (re.compile(r"(?<!\w)N/U(?!\w)", re.I), "NIPPLE UP"),
    (re.compile(r"(?<!\w)N/D(?!\w)", re.I), "NIPPLE DOWN"),
    (re.compile(r"(?<!\w)P/U(?!\w)", re.I), "PICK UP"),
    (re.compile(r"(?<!\w)B/O(?!\w)", re.I), "BREAK OUT"),
    (re.compile(r"(?<!\w)R/U(?!\w)", re.I), "RIG UP"),
    (re.compile(r"(?<!\w)L/D(?!\w)", re.I), "LAY DOWN"),
    (re.compile(r"(?<!\w)P/T(?!\w)", re.I), "PRESSURE TEST"),
    (re.compile(r"(?<!\w)R/D(?!\w)", re.I), "RIG DOWN"),
    (re.compile(r"(?<!\w)B/U(?!\w)", re.I), "BACK UP"),
]

_RE_RN_TAG      = re.compile(r"\[R-\d+\]")
_RE_DASH_FRAC   = re.compile(r"(\d)-(\d+/\d+)")
_RE_MULTI_SPACE = re.compile(r"\s+")


def clean_operation(text: str) -> str:
    """Apply the full 8-step cleaning pipeline to a single OPERATION string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text

    # 1. Strip whitespace
    s = s.strip()

    # 2. Remove [R-N] tags
    s = _RE_RN_TAG.sub("", s)

    # 3. Extract primary operation (first line only)
    s = s.split("\n")[0].strip()

    # 4. Normalize case → UPPER
    s = s.upper()

    # 5. Collapse multiple spaces
    s = _RE_MULTI_SPACE.sub(" ", s).strip()

    # 6. Strip trailing period(s)
    s = s.rstrip(".")

    # 7. Normalize fractions
    for uf, tf in UNICODE_FRACS.items():
        s = s.replace(uf.upper(), tf).replace(uf, tf)
    s = _RE_DASH_FRAC.sub(r"\1 \2", s)

    # 8. Standardize abbreviations
    for pattern, replacement in ABBREVS:
        s = pattern.sub(replacement, s)

    # Final tidy
    s = _RE_MULTI_SPACE.sub(" ", s).strip()
    return s


# ──────────────────────────────────────────────────────────────────────
# Resumable incremental writer
# ──────────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    """Count data lines (excluding header) already in output file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def main():
    parser = argparse.ArgumentParser(description="Clean OPERATION text with progress & resume")
    parser.add_argument("--input",  default="IndusTSLM/data/unique_operations_raw.csv",
                        help="Path to raw CSV (OPERATION, CODE, SUBCODE)")
    parser.add_argument("--output", default="IndusTSLM/data/unique_operations_cleaned.csv",
                        help="Path to output CSV (appends OPERATION_CLEAN)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Flush to disk every N rows (default: 5000)")
    args = parser.parse_args()

    INPUT  = args.input
    OUTPUT = args.output
    FLUSH_EVERY = args.batch_size

    # --- Read input ---
    print(f"Reading {INPUT} ...")
    with open(INPUT, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_header = next(reader)
        in_rows = list(reader)
    total = len(in_rows)
    print(f"Total rows: {total:,}")

    # --- Check how many are already done (resume support) ---
    done = count_lines(OUTPUT)
    if done > 0:
        print(f"Resuming — {done:,} rows already written, {total - done:,} remaining")
    else:
        print("Starting fresh")

    # --- Determine column indices ---
    op_idx = in_header.index("OPERATION")
    out_header = in_header + ["OPERATION_CLEAN"]

    # --- Open output for writing (or appending) ---
    if done == 0:
        # Fresh start: write header
        out_f = open(OUTPUT, "w", newline="", encoding="utf-8")
        writer = csv.writer(out_f)
        writer.writerow(out_header)
    else:
        # Resume: append mode, no header
        out_f = open(OUTPUT, "a", newline="", encoding="utf-8")
        writer = csv.writer(out_f)

    # --- Process with progress bar ---
    batch = []
    try:
        for i in tqdm(range(done, total), initial=done, total=total,
                       desc="Cleaning", unit="row", ncols=100,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            row = in_rows[i]
            op_text = row[op_idx]
            cleaned = clean_operation(op_text)
            batch.append(row + [cleaned])

            # Flush batch to disk
            if len(batch) >= FLUSH_EVERY:
                writer.writerows(batch)
                out_f.flush()
                batch = []

        # Write remaining
        if batch:
            writer.writerows(batch)
            out_f.flush()

    except (KeyboardInterrupt, Exception) as exc:
        # On interrupt or error, flush whatever we have so far
        if batch:
            writer.writerows(batch)
            out_f.flush()
        out_f.close()
        written_now = count_lines(OUTPUT) - done if done else count_lines(OUTPUT)
        print(f"\n⚠ Interrupted after writing {written_now:,} new rows (total on disk: {count_lines(OUTPUT):,}/{total:,})")
        print(f"  Re-run the same command to resume from where it stopped.")
        if isinstance(exc, KeyboardInterrupt):
            sys.exit(1)
        raise
    else:
        out_f.close()
        final_count = count_lines(OUTPUT)
        print(f"\nDone! {final_count:,} rows written to {OUTPUT}")
        print(f"Columns: {out_header}")


if __name__ == "__main__":
    main()
