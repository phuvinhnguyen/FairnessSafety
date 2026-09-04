"""Run one YAML experiment with vLLM and write one JSONL per model."""

import argparse
import gc
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from .data import load_records
from .experiments import get_experiment


def load_config(filename):
    import yaml

    path = Path(filename).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("models", "experiment", "data", "generation", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
    for key in ("path", "cache_dir"):
        if config["data"].get(key):
            config["data"][key] = str((path.parent / config["data"][key]).resolve())
    config["output_dir"] = str((path.parent / config["output_dir"]).resolve())
    return config


def parse_choice(text, labels):
    text = text.strip()
    try:
        value = json.loads(text)
        choice = value.get("choice", "").strip().upper()
    except (json.JSONDecodeError, AttributeError):
        match = re.fullmatch(r"(?:candidate\s+)?([A-Z])\.?", text, re.IGNORECASE)
        choice = match.group(1).upper() if match else ""
    return choice if choice in labels else None


def build_trials(records, config):
    settings = config["experiment"]
    size = int(settings["group_size"])
    if not 2 <= size <= 26 or len(records) < size:
        raise ValueError("group_size must be between 2 and min(26, dataset size)")
    rng = random.Random(int(settings.get("seed", 0)))
    prompt_function = get_experiment(settings["name"])
    trials = []
    for group_id in range(int(settings["num_groups"])):
        group = rng.sample(records, size)
        for permutation_id in range(int(settings.get("permutations_per_group", 1))):
            ordered = list(group)
            if permutation_id:
                rng.shuffle(ordered)
            samples = [dict(row, candidate_label=chr(65 + i)) for i, row in enumerate(ordered)]
            labels = {sample["candidate_label"]: sample["id"] for sample in samples}
            trials.append({
                "experiment_id": f"{settings['name']}:g{group_id}:p{permutation_id}",
                "experiment": settings["name"],
                "group_id": group_id,
                "permutation_id": permutation_id,
                "seed": settings.get("seed", 0),
                "candidate_order": list(labels.values()),
                "candidate_labels": labels,
                "candidates": samples,
                "prompt": prompt_function(samples),
            })
    return trials


def run(filename):
    from vllm import LLM, SamplingParams

    config = load_config(filename)
    records = load_records(config["data"])
    trials = build_trials(records, config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    repetitions = int(config["experiment"].get("generations_per_prompt", 1))

    for model in config["models"]:
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model["name"])
        output = output_dir / f"{model_name}.jsonl"
        if output.exists():
            print(f"SKIP completed model: {model['name']}")
            continue
        partial = output.with_suffix(".jsonl.partial")
        model_args = {key: value for key, value in model.items()
                      if key not in {"name", "path", "chat_template"}}
        llm = LLM(model=model["path"], **model_args)
        params = SamplingParams(n=repetitions, **config["generation"])
        prompts = [trial["prompt"] for trial in trials]
        if model.get("chat_template", False):
            tokenizer = llm.get_tokenizer()
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            ) for prompt in prompts]
        responses = llm.generate(prompts, params)
        if len(responses) != len(trials):
            raise RuntimeError("vLLM returned a different number of outputs than prompts")

        with partial.open("w", encoding="utf-8") as file:
            for trial, model_prompt, response in zip(trials, prompts, responses):
                for generation_id, candidate in enumerate(response.outputs):
                    choice = parse_choice(candidate.text, set(trial["candidate_labels"]))
                    row = dict(trial)
                    row.update({
                        "model": model["name"],
                        "model_path": model["path"],
                        "model_prompt": model_prompt,
                        "generation": config["generation"],
                        "generation_id": generation_id,
                        "raw_response": candidate.text,
                        "selected_label": choice,
                        "selected_person_id": trial["candidate_labels"].get(choice),
                        "parse_error": None if choice else "invalid_or_ambiguous_response",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        partial.replace(output)
        print(output)
        del llm
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="experiment YAML")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
