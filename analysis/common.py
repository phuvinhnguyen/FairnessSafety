"""Compact, cached access to the nested experiment JSONL outputs."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CACHE = Path(__file__).resolve().parent / ".cache"

EXPERIMENT_ORDER = [
    "description_only",
    "description_name",
    "description_gender",
    "description_race",
    "description_age",
    "demographics_without_name",
    "full_information",
    "full_information_fairness_instruction",
]

EXPERIMENT_LABELS = {
    "description_only": "01 baseline: description only",
    "description_name": "02 + name",
    "description_gender": "03 + gender",
    "description_race": "04 + race",
    "description_age": "05 + age",
    "demographics_without_name": "06 all demographics, no name",
    "full_information": "07 full information",
    "full_information_fairness_instruction": "08 full info + fairness instruction",
}


def _clean(value, unknown="Unknown"):
    if value is None or not str(value).strip() or str(value).lower() == "unknown":
        return unknown
    return str(value).strip()


def _age(value):
    match = re.search(r"\d{1,3}", str(value))
    return float(match.group()) if match else np.nan


def result_files(outputs=OUTPUTS):
    return sorted(Path(outputs).glob("*/*.jsonl"))


def _fingerprint(files):
    state = [(str(path.relative_to(ROOT)), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def load_results(refresh=False):
    """Return compact (decisions, candidates) DataFrames; rebuild when outputs change."""
    files = result_files()
    if not files:
        raise FileNotFoundError(f"No JSONL results found under {OUTPUTS}")

    fingerprint = _fingerprint(files)
    decisions_path = CACHE / "decisions.parquet"
    candidates_path = CACHE / "candidates.parquet"
    metadata_path = CACHE / "metadata.json"
    if not refresh and decisions_path.exists() and candidates_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint:
            return pd.read_parquet(decisions_path), pd.read_parquet(candidates_path)

    decision_rows, candidate_rows = [], []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                selected_id = row.get("selected_person_id")
                candidates = row.get("candidates", [])
                selected = next((item for item in candidates if item.get("id") == selected_id), {})
                order = row.get("candidate_order", [])
                position = order.index(selected_id) + 1 if selected_id in order else np.nan
                trial_key = (
                    f"{row.get('model')}|{row.get('experiment')}|g{row.get('group_id')}|"
                    f"p{row.get('permutation_id')}|r{row.get('generation_id', 0)}"
                )
                valid = selected_id is not None
                decision_rows.append({
                    "trial_key": trial_key,
                    "experiment": row.get("experiment"),
                    "experiment_label": EXPERIMENT_LABELS.get(row.get("experiment"), row.get("experiment")),
                    "model": row.get("model"),
                    "group_id": row.get("group_id"),
                    "permutation_id": row.get("permutation_id"),
                    "generation_id": row.get("generation_id", 0),
                    "selected_label": row.get("selected_label"),
                    "selected_person_id": selected_id,
                    "selected_position": position,
                    "selected_name": _clean(selected.get("name")),
                    "selected_gender": _clean(selected.get("gender")).lower().capitalize(),
                    "selected_race": _clean(selected.get("race")),
                    "selected_age": _age(selected.get("age")),
                    "valid": valid,
                    "parse_error": row.get("parse_error"),
                    "response_chars": len(row.get("raw_response", "")),
                    "output_file": str(path.relative_to(ROOT)),
                    "line_number": line_number,
                })
                for candidate_position, candidate in enumerate(candidates, 1):
                    candidate_rows.append({
                        "trial_key": trial_key,
                        "experiment": row.get("experiment"),
                        "experiment_label": EXPERIMENT_LABELS.get(row.get("experiment"), row.get("experiment")),
                        "model": row.get("model"),
                        "group_id": row.get("group_id"),
                        "permutation_id": row.get("permutation_id"),
                        "generation_id": row.get("generation_id", 0),
                        "candidate_id": candidate.get("id"),
                        "candidate_position": candidate_position,
                        "gender": _clean(candidate.get("gender")).lower().capitalize(),
                        "race": _clean(candidate.get("race")),
                        "age": _age(candidate.get("age")),
                        "selected": valid and candidate.get("id") == selected_id,
                        "valid_decision": valid,
                    })

    decisions = pd.DataFrame(decision_rows)
    candidates = pd.DataFrame(candidate_rows)
    age_bins = [0, 17, 29, 44, 59, np.inf]
    age_labels = ["0–17", "18–29", "30–44", "45–59", "60+"]
    decisions["selected_age_group"] = pd.cut(decisions["selected_age"], age_bins, labels=age_labels)
    candidates["age_group"] = pd.cut(candidates["age"], age_bins, labels=age_labels)

    CACHE.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(decisions_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "files": len(files)}, indent=2))
    return decisions, candidates


def holm_adjust(pvalues):
    """Holm family-wise error correction without an extra dependency."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def experiment_categories(frame):
    return pd.Categorical(frame, categories=EXPERIMENT_ORDER, ordered=True)


def raw_examples(decisions, limit=12):
    """Retrieve selected raw JSONL rows without retaining every prompt in memory."""
    wanted = decisions.head(limit)[["output_file", "line_number"]]
    rows = []
    for output_file, group in wanted.groupby("output_file", sort=False):
        line_numbers = set(group["line_number"])
        with (ROOT / output_file).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number in line_numbers:
                    row = json.loads(line)
                    rows.append({
                        "model": row.get("model"),
                        "experiment": row.get("experiment"),
                        "experiment_id": row.get("experiment_id"),
                        "raw_response": row.get("raw_response"),
                        "parse_error": row.get("parse_error"),
                    })
    return pd.DataFrame(rows)
