from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Closed-class English functional words and common manga dialogue vocabulary
DEFAULT_COMMON_WORDS: Set[str] = {
    # Pronouns & determiners
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "them", "us", "who", "whom", "whose", "which", "what", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    # Prepositions & conjunctions
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "than",
    # Auxiliary verbs & modals
    "be", "is", "am", "are", "was", "were", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "can", "will", "shall", "must", "might", "may",
    # Adverbs & question words
    "not", "so", "very", "too", "just", "now", "here", "there", "when", "where",
    "why", "how", "well", "only", "also", "even", "still", "back",
    # Common conversational dialogue words in manga
    "yes", "yeah", "nope", "huh", "wait", "okay", "ok", "sure", "fine",
    "good", "bad", "great", "look", "see", "come", "go", "get", "got", "give",
    "say", "said", "tell", "think", "know", "man", "girl", "boy", "guy",
    "time", "day", "way", "thing", "things", "right", "left", "please", "sorry",
    "thank", "thanks", "hello", "hey", "bye", "cool", "new", "normal", "real",
    # Common compounds that should never be split into pieces
    "together", "today", "tonight", "tomorrow", "another", "understand",
    "someone", "something", "anything", "nothing", "everyone", "everything",
    "always", "already", "almost", "without", "within", "cannot", "myself",
    "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "somewhere", "anywhere", "nowhere", "everywhere", "somebody", "anybody",
    "nobody", "everybody", "into", "onto", "upon", "maybe", "sometimes"
}


class WordSegmenter:
    """Unigram Viterbi dynamic programming word segmenter for manga OCR text.
    
    RapidOCR frequently concatenates adjacent English words into single character
    strings (e.g. 'socool' -> 'so cool', 'thenew' -> 'the new', 'backinthe' -> 'back in the')
    due to tight vertical bounding crops in manga speech bubbles.
    
    This class recovers word boundaries using:
    1. Punctuation spacing: ensuring space after punctuation if immediately followed by a letter.
    2. Contraction preservation: keeping words with apostrophes ('don't', 'it's', 'I'm') intact.
    3. Known-word protection: preventing valid single words (e.g. 'together', 'today', 'understand')
       from being broken apart.
    4. Viterbi unigram dynamic programming: finding the optimal split path that minimizes negative
       log-likelihood cost with a split penalty.
    """

    def __init__(
        self,
        vocabulary: Optional[Dict[str, int]] = None,
        split_penalty: float = 2.5,
        min_word_length_to_split: int = 5,
        max_subword_length: int = 15,
    ) -> None:
        self.vocabulary: Dict[str, int] = dict(vocabulary) if vocabulary else {}
        self.split_penalty = split_penalty
        self.min_word_length_to_split = min_word_length_to_split
        self.max_subword_length = max_subword_length
        self._total_words: int = sum(self.vocabulary.values()) if self.vocabulary else 0

    @property
    def total_words(self) -> int:
        return self._total_words

    def fit_from_token_counts(
        self,
        token_counts: Dict[str, int],
        include_common_words: bool = True,
        common_word_prior_count: int = 100,
    ) -> WordSegmenter:
        """Fits vocabulary from token frequency counts.
        
        Args:
            token_counts: Frequency dictionary of lowercased alphabetic tokens.
            include_common_words: If True, incorporates DEFAULT_COMMON_WORDS with a prior count.
            common_word_prior_count: Prior count added to each common word.
        """
        vocab = Counter()
        for tok, cnt in token_counts.items():
            clean_tok = tok.lower().strip()
            if clean_tok.isalpha() and len(clean_tok) > 0:
                vocab[clean_tok] += cnt

        if include_common_words:
            for w in DEFAULT_COMMON_WORDS:
                vocab[w] += common_word_prior_count

        self.vocabulary = dict(vocab)
        self._total_words = sum(self.vocabulary.values())
        return self

    def fit_from_training_sequences(
        self,
        golds_dict: Dict[str, List[List[Dict[str, str]]]],
        allowed_train_ids: Set[str],
        forbidden_val_test_ids: Optional[Set[str]] = None,
        include_common_words: bool = True,
        common_word_prior_count: int = 100,
    ) -> WordSegmenter:
        """Fits vocabulary strictly from permitted training sequences.
        
        Strict Leakage Enforcement:
        - Only sequences in `allowed_train_ids` are read.
        - Raises ValueError if any sequence in `forbidden_val_test_ids` is present in training data.
        """
        if forbidden_val_test_ids:
            overlap = allowed_train_ids.intersection(forbidden_val_test_ids)
            if overlap:
                raise ValueError(
                    f"Data leakage detected! Training sequence IDs overlap with validation/test IDs: {overlap}"
                )

        token_counts = Counter()
        for seq_id in allowed_train_ids:
            if seq_id not in golds_dict:
                continue
            pages = golds_dict[seq_id]
            for page in pages:
                for item in page:
                    text = item.get("text", "")
                    # Extract words using regex
                    words = re.findall(r"[A-Za-z]+", text)
                    for w in words:
                        token_counts[w.lower()] += 1

        return self.fit_from_token_counts(
            token_counts=token_counts,
            include_common_words=include_common_words,
            common_word_prior_count=common_word_prior_count,
        )

    def save_to_json(self, path: Path | str) -> None:
        """Serializes segmenter vocabulary and parameters to JSON."""
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "split_penalty": self.split_penalty,
            "min_word_length_to_split": self.min_word_length_to_split,
            "max_subword_length": self.max_subword_length,
            "vocabulary_size": len(self.vocabulary),
            "total_words": self._total_words,
            "vocabulary": self.vocabulary,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load_from_json(cls, path: Path | str) -> WordSegmenter:
        """Loads segmenter from serialized JSON file."""
        target_path = Path(path).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"Segmenter model file not found at {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segmenter = cls(
            vocabulary=data.get("vocabulary", {}),
            split_penalty=data.get("split_penalty", 2.5),
            min_word_length_to_split=data.get("min_word_length_to_split", 5),
            max_subword_length=data.get("max_subword_length", 15),
        )
        return segmenter

    def segment_token(self, chunk: str) -> str:
        """Segments a single alphabetic chunk using Viterbi unigram dynamic programming."""
        if not chunk or len(chunk) < self.min_word_length_to_split or not chunk.isalpha():
            return chunk

        # Do not split contractions or apostrophes
        if "'" in chunk:
            return chunk

        lower = chunk.lower()

        # If the word itself is already in the vocabulary, do NOT split
        if lower in self.vocabulary and self.vocabulary[lower] >= 1:
            return chunk

        n = len(lower)
        cost = [0.0] + [float("inf")] * n
        split_idx = [0] * (n + 1)

        total = self._total_words if self._total_words > 0 else 1

        for j in range(1, n + 1):
            min_i = max(0, j - self.max_subword_length)
            for i in range(min_i, j):
                subword = lower[i:j]
                if subword in self.vocabulary:
                    freq = self.vocabulary[subword]
                    penalty = self.split_penalty if i > 0 else 0.0
                    c = cost[i] + (-math.log(freq / total)) + penalty
                    if c < cost[j]:
                        cost[j] = c
                        split_idx[j] = i

        # If a valid split path covering the entire token was found
        if cost[n] < float("inf") and split_idx[n] != 0:
            idx = n
            splits_pos: List[int] = []
            while idx > 0:
                splits_pos.append(idx)
                idx = split_idx[idx]
            splits_pos.reverse()

            # Reconstruct string preserving original casing
            parts: List[str] = []
            curr = 0
            for sp in splits_pos:
                parts.append(chunk[curr:sp])
                curr = sp
            return " ".join(parts)

        return chunk

    def segment_text(self, text: str) -> str:
        """Processes a full dialogue line, adding punctuation spacing and segmenting glued words."""
        if not text or not text.strip():
            return text

        # 1. Punctuation spacing: ensure space after punctuation if immediately followed by letter
        # e.g. 'HUH?DOES' -> 'HUH? DOES', 'WAIT,LOOK' -> 'WAIT, LOOK'
        formatted = re.sub(r"([?!,.:;])([A-Za-z])", r"\1 \2", text)

        # 2. Tokenize into words (including contractions like don't, I'm) and non-word spans
        parts: List[str] = []
        tokens_raw = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+", formatted)
        for chunk in tokens_raw:
            if chunk.isalpha():
                parts.append(self.segment_token(chunk))
            else:
                parts.append(chunk)

        result = "".join(parts)
        # Clean up any double spaces that may have been created
        result = re.sub(r" +", " ", result).strip()
        return result
