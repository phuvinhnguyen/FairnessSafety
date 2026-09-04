"""Dataset loading and conservative schema normalization."""

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ALIASES = {
    "id": ("id", "person_id", "case_number", "case_id", "missing_person_id", "canonical_url"),
    "name": ("name", "full_name", "fullname", "missing_person"),
    "age": ("age", "current_age", "age_at_disappearance", "missing_age"),
    "gender": ("gender", "sex"),
    "race": ("race", "ethnicity", "race_ethnicity"),
    "description": ("description", "physical_description", "distinguishing_characteristics", "details", "remarks"),
    "location": ("location", "missing_location", "last_seen_location", "missing_from", "city"),
    "circumstances": ("circumstances", "case_details", "missing_details", "details_of_disappearance", "narrative"),
    "missing_date": ("missing_date", "date_missing", "last_seen_date", "missing_since"),
    "classification": ("classification", "case_type"),
    "medical_conditions": ("medical_conditions", "medical_condition"),
    "height_weight": ("height_and_weight", "height_weight"),
    "clothing": ("clothing_jewelry_description", "clothing_description", "clothing"),
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _first(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = row.get(alias)
        if value is not None and str(value).strip():
            return value
    return None


def sanitize_description(text: str, name: str, race: str) -> str:
    clean = text or ""
    if name and name != "unknown":
        parts = re.findall(r"[A-Za-z]+", name)
        if parts:
            flexible_name = r"\b" + r"\W*".join(map(re.escape, parts)) + r"\b"
            clean = re.sub(flexible_name, "[name removed]", clean, flags=re.IGNORECASE)
            for part in parts:
                if len(part) > 2:
                    clean = re.sub(rf"\b{re.escape(part)}\b", "[name removed]", clean, flags=re.IGNORECASE)
            initials = [part for part in parts if len(part) == 1]
            if len(initials) >= 2:
                pattern = r"\b" + r"\W*".join(map(re.escape, initials)) + r"\b"
                clean = re.sub(pattern, "[name removed]", clean, flags=re.IGNORECASE)
    terms = [
        "male", "female", "man", "woman", "boy", "girl", "he", "she", "him", "her", "his", "hers",
        "husband", "wife", "father", "mother", "son", "daughter", "brother", "sister", "gentleman", "lady",
    ]
    if race and race != "unknown":
        terms.extend(str(race).split())
    clean = re.sub(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", "[demographic removed]", clean,
                   flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:aged?|age)\s*[:=-]?\s*\d{1,3}\b|\b\d{1,3}[ -]years?[ -]old\b",
                   "[age removed]", clean, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean).strip()


def normalize_row(raw: dict[str, Any], index: int) -> dict[str, Any]:
    if raw.get("_schema_version") == 1:
        normalized = dict(raw)
        normalized["source_index"] = index
        return normalized
    row = {_key(str(key)): value for key, value in raw.items()}
    normalized = {field: _first(row, aliases) for field, aliases in ALIASES.items()}
    normalized = {key: ("unknown" if value is None or str(value).strip() == "" else value)
                  for key, value in normalized.items()}
    if normalized["id"] == "unknown":
        fingerprint = json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
        normalized["id"] = "person-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    normalized["id"] = str(normalized["id"])
    age_match = re.search(r"\d{1,3}", str(normalized["age"]))
    normalized["age"] = int(age_match.group()) if age_match else "unknown"
    normalized["gender"] = str(normalized["gender"]).strip().lower()
    physical_parts = [normalized[key] for key in ("description", "height_weight", "clothing")
                      if normalized[key] != "unknown"]
    normalized["description"] = " ".join(map(str, physical_parts)) or "unknown"
    normalized["description_demographic_sanitized"] = sanitize_description(
        str(normalized["description"]), str(normalized["name"]), str(normalized["race"])
    )
    normalized["circumstances_demographic_sanitized"] = sanitize_description(
        str(normalized["circumstances"]), str(normalized["name"]), str(normalized["race"])
    )
    normalized["source_index"] = index
    normalized["_schema_version"] = 1
    return normalized


def _local_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        yield from data
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        raise ValueError(f"Unsupported data file: {path}; use .jsonl, .json, or .csv")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config.get("source", "jsonl")
    if source == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Hugging Face input requires: pip install -e '.[data]'") from exc
        dataset = load_dataset(
            config["dataset"], config.get("subset"), split=config.get("split", "train"),
            cache_dir=config.get("cache_dir"),
        )
        raw_rows = (dict(row) for row in dataset)
    else:
        raw_rows = _local_rows(Path(config["path"]))

    records = [normalize_row(row, index) for index, row in enumerate(raw_rows)]
    cache_path = config.get("cache_path")
    if cache_path:
        write_jsonl(records, cache_path)
    return records
