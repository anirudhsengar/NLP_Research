from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

from ai_text_detector.icnale import load_icnale
from ai_text_detector.schema import LABEL_AI, LABEL_HUMAN, ensure_canonical, write_jsonl
from ai_text_detector.splitting import split_by_group


def prepare_all(
    config: dict[str, Any],
    *,
    max_hc3_samples: int | None = None,
    max_gptwiki_samples: int | None = None,
    include_optional_icnale: bool | None = None,
    skip_icnale: bool = False,
) -> dict[str, Path]:
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    hc3 = load_hc3(config, max_samples=max_hc3_samples)
    gptwiki = load_gpt_wiki_intro(config, max_samples=max_gptwiki_samples)
    outputs = {
        "hc3": write_jsonl(hc3, processed_dir / "hc3.jsonl"),
        "gpt_wiki_intro": write_jsonl(gptwiki, processed_dir / "gpt_wiki_intro.jsonl"),
    }
    if not skip_icnale:
        icnale = load_icnale(config, include_optional=include_optional_icnale)
        outputs["icnale"] = write_jsonl(icnale, processed_dir / "icnale_fairness.jsonl")
    return outputs


def load_hc3(config: dict[str, Any], *, max_samples: int | None = None) -> pd.DataFrame:
    ds_cfg = config["datasets"]["hc3"]
    try:
        split_iter = list(_iter_dataset_splits(_load_hf_dataset(ds_cfg["name"], ds_cfg.get("configs"))))
    except RuntimeError:
        split_iter = _load_hc3_jsonl_splits(ds_cfg["name"], ds_cfg.get("configs"))
    rows = []
    for split_name, split in split_iter:
        for idx, record in enumerate(split):
            source = _first_nonempty(record, "source", "domain", "task", default=split_name)
            group_id = f"hc3:{source}:{split_name}:{idx}"
            question = str(record.get("question", "") or record.get("prompt", "") or "")
            for answer_idx, answer in enumerate(_iter_answers(record.get("human_answers"))):
                rows.append(
                    _row(
                        sample_id=f"{group_id}:human:{answer_idx}",
                        dataset="hc3",
                        split="",
                        text=answer,
                        label=LABEL_HUMAN,
                        source=source,
                        domain=source,
                        group_id=group_id,
                        license_tag="HC3 source-dependent",
                    )
                )
            for answer_idx, answer in enumerate(_iter_answers(record.get("chatgpt_answers"))):
                rows.append(
                    _row(
                        sample_id=f"{group_id}:ai:{answer_idx}",
                        dataset="hc3",
                        split="",
                        text=answer,
                        label=LABEL_AI,
                        source=source,
                        domain=source,
                        group_id=group_id,
                        license_tag="HC3 source-dependent",
                    )
                )
            if not record.get("human_answers") and not record.get("chatgpt_answers"):
                rows.extend(_fallback_binary_rows(record, group_id, source, question))

    df = ensure_canonical(pd.DataFrame(rows))
    if max_samples:
        df = _balanced_sample(df, max_samples=max_samples, seed=config["random_seed"])
    split_cfg = config["splits"]
    return split_by_group(
        df,
        train_size=split_cfg["train_size"],
        validation_size=split_cfg["validation_size"],
        test_size=split_cfg["test_size"],
        seed=config["random_seed"],
    )


def load_gpt_wiki_intro(config: dict[str, Any], *, max_samples: int | None = None) -> pd.DataFrame:
    ds_cfg = config["datasets"]["gpt_wiki_intro"]
    try:
        split_iter = list(_iter_dataset_splits(load_dataset(ds_cfg["name"])))
    except Exception:
        split_iter = _load_gpt_wiki_csv_splits(ds_cfg["name"])
    rows = []
    for split_name, split in split_iter:
        for idx, record in enumerate(split):
            group_id = f"gptwiki:{split_name}:{idx}"
            human_text = _first_nonempty(record, "wiki_intro", "human_intro", "human", default="")
            ai_text = _first_nonempty(record, "generated_intro", "gpt_intro", "generated", default="")
            if human_text:
                rows.append(
                    _row(
                        sample_id=f"{group_id}:human",
                        dataset="gpt_wiki_intro",
                        split="ood",
                        text=human_text,
                        label=LABEL_HUMAN,
                        source="wikipedia",
                        domain="wikipedia_intro",
                        group_id=group_id,
                        license_tag="GPT-wiki-intro source-dependent",
                    )
                )
            if ai_text:
                rows.append(
                    _row(
                        sample_id=f"{group_id}:ai",
                        dataset="gpt_wiki_intro",
                        split="ood",
                        text=ai_text,
                        label=LABEL_AI,
                        source="gpt_generated",
                        domain="wikipedia_intro",
                        group_id=group_id,
                        license_tag="GPT-wiki-intro source-dependent",
                    )
                )

    df = ensure_canonical(pd.DataFrame(rows))
    if max_samples:
        df = _balanced_sample(df, max_samples=max_samples, seed=config["random_seed"])
    return df


def _load_hf_dataset(name: str, configs: list[str] | None) -> DatasetDict:
    tried = []
    candidate_configs = list(configs or [])
    if not candidate_configs:
        try:
            candidate_configs = get_dataset_config_names(name)
        except Exception:
            candidate_configs = [None]
    if None not in candidate_configs:
        candidate_configs.append(None)

    last_error: Exception | None = None
    for config_name in candidate_configs:
        tried.append(config_name)
        try:
            return load_dataset(name, config_name) if config_name else load_dataset(name)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load Hugging Face dataset {name}; tried configs {tried}") from last_error


def _load_hc3_jsonl_splits(name: str, configs: list[str] | None) -> list[tuple[str, list[dict]]]:
    selected = configs or ["all"]
    splits = []
    for config_name in selected:
        filename = f"{config_name}.jsonl"
        local_path = hf_hub_download(repo_id=name, repo_type="dataset", filename=filename)
        records = pd.read_json(local_path, lines=True).to_dict(orient="records")
        splits.append((str(config_name), records))
    return splits


def _load_gpt_wiki_csv_splits(name: str) -> list[tuple[str, list[dict]]]:
    local_path = hf_hub_download(repo_id=name, repo_type="dataset", filename="GPT-wiki-intro.csv.zip")
    records = pd.read_csv(local_path, compression="zip").to_dict(orient="records")
    return [("train", records)]


def _iter_dataset_splits(dataset):
    if isinstance(dataset, DatasetDict):
        yield from dataset.items()
    else:
        yield "train", dataset


def _iter_answers(value):
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item).strip()
            if text:
                yield text


def _fallback_binary_rows(record: dict, group_id: str, source: str, question: str) -> list[dict]:
    text = _first_nonempty(record, "text", "answer", "completion", default="")
    label_value = record.get("label")
    if not text or label_value is None:
        return []
    label = LABEL_AI if str(label_value).lower() in {"1", "ai", "chatgpt", "generated"} else LABEL_HUMAN
    return [
        _row(
            sample_id=f"{group_id}:fallback",
            dataset="hc3",
            split="",
            text=text,
            label=label,
            source=source,
            domain=source,
            group_id=group_id or question,
            license_tag="HC3 source-dependent",
        )
    ]


def _row(**kwargs) -> dict:
    kwargs.setdefault("native_status", "")
    kwargs.setdefault("country", "")
    kwargs.setdefault("l1", "")
    kwargs.setdefault("cefr_level", "")
    return kwargs


def _first_nonempty(record: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _balanced_sample(df: pd.DataFrame, *, max_samples: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    per_label = max_samples // max(1, df["label"].nunique())
    sampled_indices = []
    for _, group in df.groupby("label"):
        sampled_indices.extend(group.sample(min(len(group), per_label), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices].sample(frac=1.0, random_state=seed)
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.reset_index(drop=True)
