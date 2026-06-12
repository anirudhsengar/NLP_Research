from __future__ import annotations

import re
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


class TextColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _texts(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(["text"])


class BasicStatsTransformer(BaseEstimator, TransformerMixin):
    feature_names = np.array(
        [
            "stats:char_count",
            "stats:word_count",
            "stats:sentence_count",
            "stats:avg_word_len",
            "stats:type_token_ratio",
            "stats:hapax_ratio",
            "stats:stopword_ratio",
            "stats:punct_ratio",
            "stats:digit_ratio",
            "stats:uppercase_ratio",
            "stats:comma_per_word",
            "stats:question_per_word",
            "stats:exclaim_per_word",
            "stats:semicolon_per_word",
            "stats:sentence_len_mean",
            "stats:sentence_len_std",
            "stats:sentence_len_cv",
            "stats:sentence_len_p10",
            "stats:sentence_len_p90",
            "stats:sentence_len_range",
            "stats:sentence_len_delta_mean",
            "stats:burstiness_index",
            "stats:word_len_std",
            "stats:unigram_repeat_ratio",
            "stats:word_frequency_entropy",
            "stats:flesch_reading_ease",
            "stats:flesch_kincaid_grade",
            "stats:capitalized_token_ratio",
            "stats:repeated_bigram_ratio",
        ]
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        matrix = [self._features_for_text(text) for text in _texts(X)]
        return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def _features_for_text(self, text: str) -> list[float]:
        text = str(text)
        words = WORD_RE.findall(text)
        lower_words = [word.lower() for word in words]
        word_count = len(words)
        char_count = len(text)
        sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
        sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
        sentence_count = max(1, len(sentence_lengths))
        unique_words = set(lower_words)
        counts = Counter(lower_words)
        hapax = sum(1 for count in counts.values() if count == 1)
        avg_word_len = _safe_div(sum(len(word) for word in words), word_count)
        sentence_mean = _safe_div(sum(sentence_lengths), sentence_count)
        sentence_std = float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        sentence_p10 = float(np.percentile(sentence_lengths, 10)) if sentence_lengths else 0.0
        sentence_p90 = float(np.percentile(sentence_lengths, 90)) if sentence_lengths else 0.0
        sentence_range = float(max(sentence_lengths) - min(sentence_lengths)) if sentence_lengths else 0.0
        sentence_deltas = [
            abs(sentence_lengths[idx] - sentence_lengths[idx - 1])
            for idx in range(1, len(sentence_lengths))
        ]
        unigram_repeats = word_count - len(unique_words)
        probabilities = np.asarray(list(counts.values()), dtype=np.float64)
        probabilities = probabilities / probabilities.sum() if probabilities.size else probabilities
        word_entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else 0.0
        syllables = sum(_estimate_syllables(word) for word in words)
        capitalized = sum(1 for word in words if word[:1].isupper())
        bigrams = list(zip(lower_words, lower_words[1:]))
        repeated_bigrams = len(bigrams) - len(set(bigrams))

        return [
            float(char_count),
            float(word_count),
            float(sentence_count),
            avg_word_len,
            _safe_div(len(unique_words), word_count),
            _safe_div(hapax, word_count),
            _safe_div(sum(1 for word in lower_words if word in STOPWORDS), word_count),
            _safe_div(sum(1 for char in text if char in ".,;:!?()[]{}\"'"), max(1, char_count)),
            _safe_div(sum(1 for char in text if char.isdigit()), max(1, char_count)),
            _safe_div(sum(1 for char in text if char.isupper()), max(1, char_count)),
            _safe_div(text.count(","), word_count),
            _safe_div(text.count("?"), word_count),
            _safe_div(text.count("!"), word_count),
            _safe_div(text.count(";"), word_count),
            sentence_mean,
            sentence_std,
            _safe_div(sentence_std, sentence_mean),
            sentence_p10,
            sentence_p90,
            sentence_range,
            float(np.mean(sentence_deltas)) if sentence_deltas else 0.0,
            _safe_div(sentence_std - sentence_mean, sentence_std + sentence_mean),
            float(np.std([len(word) for word in words])) if words else 0.0,
            _safe_div(unigram_repeats, word_count),
            word_entropy,
            _flesch_reading_ease(word_count, sentence_count, syllables),
            _flesch_kincaid_grade(word_count, sentence_count, syllables),
            _safe_div(capitalized, word_count),
            _safe_div(repeated_bigrams, len(bigrams)),
        ]


class SpacyStatsTransformer(BaseEstimator, TransformerMixin):
    feature_names = np.array(
        [
            "spacy:pos_noun_ratio",
            "spacy:pos_verb_ratio",
            "spacy:pos_adj_ratio",
            "spacy:pos_adv_ratio",
            "spacy:pos_pron_ratio",
            "spacy:pos_prop_ratio",
            "spacy:entity_density",
            "spacy:dependency_depth_mean",
        ]
    )

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._nlp = None

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X):
        self._load()
        docs = self._nlp.pipe(_texts(X), batch_size=self.batch_size)
        matrix = [self._features_for_doc(doc) for doc in docs]
        return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state

    def _load(self):
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("Install the spacy extra to use spaCy features: uv sync --extra spacy") from exc
        try:
            self._nlp = spacy.load(self.model_name, disable=[])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self.model_name!r} is not installed. "
                f"Run: uv run python -m spacy download {self.model_name}"
            ) from exc

    def _features_for_doc(self, doc) -> list[float]:
        tokens = [token for token in doc if not token.is_space and not token.is_punct]
        n_tokens = len(tokens)
        pos_counts = Counter(token.pos_ for token in tokens)
        depths = [_dependency_depth(token) for token in tokens]
        return [
            _safe_div(pos_counts["NOUN"], n_tokens),
            _safe_div(pos_counts["VERB"], n_tokens),
            _safe_div(pos_counts["ADJ"], n_tokens),
            _safe_div(pos_counts["ADV"], n_tokens),
            _safe_div(pos_counts["PRON"], n_tokens),
            _safe_div(pos_counts["PROPN"], n_tokens),
            _safe_div(len(doc.ents), n_tokens),
            float(np.mean(depths)) if depths else 0.0,
        ]


class LanguageModelStatsTransformer(BaseEstimator, TransformerMixin):
    """GPT-2/GLTR-style statistics used as opt-in interpretable features."""

    feature_names = np.array(
        [
            "lm:perplexity",
            "lm:mean_token_logprob",
            "lm:std_token_logprob",
            "lm:gltr_top10_ratio",
            "lm:gltr_top100_ratio",
            "lm:gltr_top1000_ratio",
            "lm:gltr_over1000_ratio",
            "lm:token_rank_mean",
            "lm:token_rank_std",
        ]
    )

    def __init__(self, model_name: str = "gpt2", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self._torch = None
        self._tokenizer = None
        self._model = None

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X):
        self._load()
        matrix = [self._features_for_text(text) for text in _texts(X)]
        return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_torch"] = None
        state["_tokenizer"] = None
        state["_model"] = None
        return state

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install transformer dependencies to use language-model features: "
                "uv sync --extra baselines"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()

    def _features_for_text(self, text: str) -> list[float]:
        encoded = self._tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=int(self.max_length),
        )
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] < 2:
            return [0.0] * len(self.feature_names)

        with self._torch.no_grad():
            logits = self._model(input_ids).logits[0, :-1, :]
            targets = input_ids[0, 1:]
            target_logits = logits.gather(1, targets.unsqueeze(1))
            log_probs = self._torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            ranks = (logits > target_logits).sum(dim=1) + 1

        log_probs_np = token_log_probs.cpu().numpy()
        ranks_np = ranks.cpu().numpy()
        mean_logprob = float(log_probs_np.mean())
        perplexity = float(math.exp(min(20.0, -mean_logprob)))
        return [
            perplexity,
            mean_logprob,
            float(log_probs_np.std()),
            float((ranks_np <= 10).mean()),
            float((ranks_np <= 100).mean()),
            float((ranks_np <= 1000).mean()),
            float((ranks_np > 1000).mean()),
            float(ranks_np.mean()),
            float(ranks_np.std()),
        ]


def make_feature_pipeline(feature_config: dict) -> FeatureUnion:
    word_range = tuple(feature_config.get("word_ngram_range", [1, 2]))
    char_range = tuple(feature_config.get("char_ngram_range", [3, 5]))
    min_df = feature_config.get("min_df", 2)
    lowercase = bool(feature_config.get("lowercase", True))

    transformers = []

    if feature_config.get("use_word_tfidf", True):
        transformers.append(
            (
                "word_tfidf",
                Pipeline(
                    [
                        ("text", TextColumn()),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=word_range,
                                min_df=min_df,
                                max_features=feature_config.get("max_word_features", 30000),
                                lowercase=lowercase,
                            ),
                        ),
                    ]
                ),
            )
        )

    if feature_config.get("use_char_tfidf", True):
        transformers.append(
            (
                "char_tfidf",
                Pipeline(
                    [
                        ("text", TextColumn()),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=char_range,
                                min_df=min_df,
                                max_features=feature_config.get("max_char_features", 30000),
                                lowercase=lowercase,
                            ),
                        ),
                    ]
                ),
            )
        )

    if feature_config.get("use_basic_stats", True):
        transformers.append(("basic_stats", BasicStatsTransformer()))

    if feature_config.get("use_spacy", False):
        transformers.append(
            ("spacy_stats", SpacyStatsTransformer(feature_config.get("spacy_model", "en_core_web_sm")))
        )

    if feature_config.get("use_lm_stats", False):
        transformers.append(
            (
                "lm_stats",
                LanguageModelStatsTransformer(
                    feature_config.get("lm_model_name", "gpt2"),
                    int(feature_config.get("lm_max_length", 256)),
                ),
            )
        )

    if not transformers:
        raise ValueError("At least one feature family must be enabled.")

    return FeatureUnion(transformer_list=transformers, n_jobs=None)


def feature_names(feature_union: FeatureUnion) -> np.ndarray:
    try:
        return feature_union.get_feature_names_out()
    except Exception:
        names: list[str] = []
        for name, transformer in feature_union.transformer_list:
            try:
                local_names = transformer.get_feature_names_out()
            except Exception:
                local_names = [name]
            names.extend([f"{name}__{local}" for local in local_names])
        return np.asarray(names, dtype=object)


def _texts(X) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return X["text"].fillna("").astype(str).tolist()
    if isinstance(X, pd.Series):
        return X.fillna("").astype(str).tolist()
    return [str(item) for item in X]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _estimate_syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(words: int, sentences: int, syllables: int) -> float:
    if words == 0:
        return 0.0
    return 206.835 - 1.015 * (words / max(1, sentences)) - 84.6 * (syllables / words)


def _flesch_kincaid_grade(words: int, sentences: int, syllables: int) -> float:
    if words == 0:
        return 0.0
    return 0.39 * (words / max(1, sentences)) + 11.8 * (syllables / words) - 15.59


def _dependency_depth(token) -> int:
    depth = 0
    current = token
    seen = set()
    while current.head is not current and current.i not in seen:
        seen.add(current.i)
        current = current.head
        depth += 1
        if depth > 50:
            break
    return depth
