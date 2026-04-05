"""Helper to prepare batches and manage progress for LLM-based operation cleaning."""
import json
import os

DATA_DIR = "/home/yahia.shaaban/project/IndusTSLM/data"
RECORDS_PATH = os.path.join(DATA_DIR, "operations_to_clean.json")
MAPPING_PATH = os.path.join(DATA_DIR, "operation_mapping.json")
BATCH_SIZE = 50

def load_records():
    with open(RECORDS_PATH) as f:
        return json.load(f)

def load_mapping():
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            return json.load(f)
    return {}

def save_mapping(mapping):
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=0)

def get_next_batches(n_agents=5):
    """Return the next n_agents batches of uncleaned operations."""
    records = load_records()
    mapping = load_mapping()
    done = len(mapping)
    total = len(records)

    batches = []
    for i in range(n_agents):
        start = done + i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        if start >= total:
            break
        batch_records = records[start:end]
        lines = []
        for r in batch_records:
            first_line = r["op"].split("\n")[0].strip() if r["op"] else ""
            lines.append(f'[CODE:{r["code"]}] [SUBCODE:{r["subcode"]}] {first_line}')
        batches.append({
            "batch_id": i,
            "start": start,
            "end": end,
            "lines": lines,
            "raw_ops": [r["op"] for r in batch_records],
        })
    return batches, done, total

def parse_and_save(batch, response_lines):
    """Parse agent response and save to mapping."""
    mapping = load_mapping()
    raw_ops = batch["raw_ops"]

    # Match response lines to raw operations by position
    for i, raw_op in enumerate(raw_ops):
        if i < len(response_lines) and response_lines[i].strip():
            mapping[raw_op] = response_lines[i].strip()
        else:
            mapping[raw_op] = ""  # mark as attempted

    save_mapping(mapping)
    return len(mapping)

if __name__ == "__main__":
    mapping = load_mapping()
    records = load_records()
    done = len(mapping)
    total = len(records)
    pct = 100 * done / total if total else 0
    bar_len = 40
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Progress: {bar} {done:,}/{total:,} ({pct:.1f}%)\n")
