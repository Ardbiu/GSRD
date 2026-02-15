from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsrd.data.datasets import get_dataset_spec
from gsrd.utils.hashing import stable_hash
from gsrd.utils.io import ensure_dir, write_json
from gsrd.vocab.taxonomy import (
    COARSE_ALIASES,
    COARSE_GROUPS,
    FINE_VARIANTS,
    build_term_to_classes,
    normalize_label,
)

LOGGER = logging.getLogger(__name__)

DISTRACTOR_TERMS = [
    "background texture",
    "road marking",
    "shadow patch",
    "window reflection",
    "tree canopy",
    "building facade",
    "camera flare",
    "pavement pattern",
    "glass glare",
    "motion blur",
]


@dataclass(frozen=True)
class VocabList:
    vocab_id: str
    granularity: str
    terms: list[str]
    class_to_term: dict[str, str]
    term_to_classes: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_id": self.vocab_id,
            "granularity": self.granularity,
            "terms": self.terms,
            "class_to_term": self.class_to_term,
            "term_to_classes": self.term_to_classes,
        }


def _standard(classes: list[str]) -> dict[str, str]:
    return {c: c for c in classes}


def _standard_lexical_variant(classes: list[str], rng: random.Random) -> dict[str, str]:
    templates = [
        "{c}",
        "a {c}",
        "the {c}",
        "{c} object",
        "{c} instance",
        "visible {c}",
        "target {c}",
    ]
    return {c: templates[rng.randrange(len(templates))].format(c=c) for c in classes}


def _coarse(classes: list[str], rng: random.Random) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cls in classes:
        coarse = COARSE_GROUPS.get(cls, "other")
        aliases = COARSE_ALIASES.get(coarse, [coarse])
        mapping[cls] = aliases[rng.randrange(len(aliases))]
    return mapping


def _fine(classes: list[str], rng: random.Random) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used_terms: set[str] = set()
    for cls in classes:
        candidates = [normalize_label(v) for v in FINE_VARIANTS.get(cls, [cls])]
        pick = candidates[rng.randrange(len(candidates))]
        if pick in used_terms:
            pick = f"{pick} ({cls})"
        used_terms.add(pick)
        mapping[cls] = pick
    return mapping


def _mixed(classes: list[str], rng: random.Random, coarse_ratio: float) -> dict[str, str]:
    mapping: dict[str, str] = {}
    classes_shuffled = classes[:]
    rng.shuffle(classes_shuffled)
    cutoff = int(len(classes_shuffled) * coarse_ratio)
    coarse_set = set(classes_shuffled[:cutoff])

    coarse_map = _coarse(classes, rng)
    fine_map = _fine(classes, rng)

    for cls in classes:
        mapping[cls] = coarse_map[cls] if cls in coarse_set else fine_map[cls]
    return mapping


def _counterfactual(
    classes: list[str],
    rng: random.Random,
    noise_ratio: float,
) -> tuple[dict[str, str], list[str]]:
    templates = [
        "occluded {c}",
        "distant {c}",
        "tiny {c}",
        "blurry {c}",
        "side-view {c}",
        "partial {c}",
    ]
    class_to_term = {c: templates[rng.randrange(len(templates))].format(c=c) for c in classes}
    n_noise = max(2, int(len(classes) * noise_ratio))
    distractors = rng.sample(DISTRACTOR_TERMS, k=min(n_noise, len(DISTRACTOR_TERMS)))
    return class_to_term, distractors


def _build_vocab_list(
    granularity: str,
    class_to_term: dict[str, str],
    extra_terms: list[str] | None = None,
) -> VocabList:
    terms = sorted(set(class_to_term.values()) | set(extra_terms or []))
    term_to_classes = build_term_to_classes(class_to_term)
    for term in extra_terms or []:
        term_to_classes.setdefault(term, [])
    vocab_id = stable_hash(
        {
            "granularity": granularity,
            "class_to_term": class_to_term,
            "terms": terms,
        }
    )
    return VocabList(
        vocab_id=vocab_id,
        granularity=granularity,
        terms=terms,
        class_to_term=dict(sorted(class_to_term.items())),
        term_to_classes=term_to_classes,
    )


def generate_vocabularies(
    cache_root: Path,
    dataset_name: str,
    lists_per_granularity: int,
    seed: int,
    mixed_ratio: float,
    include_counterfactual: bool = True,
    counterfactual_lists: int = 3,
    counterfactual_noise_ratio: float = 0.2,
) -> Path:
    spec = get_dataset_spec(cache_root, dataset_name)
    classes = [normalize_label(name) for _, name in sorted(spec.category_id_to_name.items())]
    rng = random.Random(seed)

    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "lists_per_granularity": lists_per_granularity,
        "granularities": {"standard": [], "coarse": [], "fine": [], "mixed": []},
    }
    if include_counterfactual:
        payload["granularities"]["counterfactual"] = []

    def fill_unique(granularity: str, target_lists: int, builder) -> None:
        seen_ids: set[str] = set()
        attempts = 0
        max_attempts = max(50, target_lists * 60)
        while len(payload["granularities"][granularity]) < target_lists and attempts < max_attempts:
            attempts += 1
            built = builder()
            if isinstance(built, tuple):
                class_to_term, extra_terms = built
            else:
                class_to_term, extra_terms = built, []
            vocab = _build_vocab_list(granularity, class_to_term, extra_terms=extra_terms).to_dict()
            if vocab["vocab_id"] in seen_ids:
                continue
            seen_ids.add(vocab["vocab_id"])
            payload["granularities"][granularity].append(vocab)
        if len(payload["granularities"][granularity]) < target_lists:
            LOGGER.warning(
                "Only generated %d/%d unique vocab lists for granularity=%s on dataset=%s",
                len(payload["granularities"][granularity]),
                target_lists,
                granularity,
                dataset_name,
            )

    fill_unique("standard", lists_per_granularity, lambda: _standard_lexical_variant(classes, rng))
    fill_unique("coarse", lists_per_granularity, lambda: _coarse(classes, rng))
    fill_unique("fine", lists_per_granularity, lambda: _fine(classes, rng))
    fill_unique("mixed", lists_per_granularity, lambda: _mixed(classes, rng, coarse_ratio=mixed_ratio))
    if include_counterfactual:
        fill_unique(
            "counterfactual",
            max(1, counterfactual_lists),
            lambda: _counterfactual(classes, rng, noise_ratio=counterfactual_noise_ratio),
        )

    out_dir = ensure_dir(cache_root / "vocabs")
    out_path = out_dir / f"{dataset_name}_vocabs.json"
    write_json(out_path, payload)
    LOGGER.info("Generated vocabularies for %s -> %s", dataset_name, out_path)
    return out_path
