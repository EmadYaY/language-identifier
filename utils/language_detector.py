"""
Language Detection Utility Classes
File: utils/language_detector.py

This module contains all the core classes for language identification.
Based on: Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization.

Author: Emad
Version: 1.0.0
"""

import numpy as np
from collections import Counter

# Minimum number of n-grams required before issuing a low-confidence warning
MIN_NGRAMS_FOR_CONFIDENCE = 5


def extract_trigrams(text):
    """
    Extract character tri-grams from text.

    Args:
        text (str): Input text

    Returns:
        list: List of tri-grams

    Example:
        >>> extract_trigrams("hello")
        ['_he', 'hel', 'ell', 'llo', 'lo_']
    """
    text = text.lower()
    text = '_' + text + '_'
    trigrams = []
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        trigrams.append(trigram)
    return trigrams


def compute_trigram_frequencies(text):
    """
    Compute frequency distribution of tri-grams.

    Args:
        text (str): Input text

    Returns:
        Counter: Frequency distribution of tri-grams
    """
    trigrams = extract_trigrams(text)
    return Counter(trigrams)


class LanguageModel:
    """
    Tri-gram language model for a single language.

    Attributes:
        language (str): Name of the language
        trigram_counts (Counter): Frequency counts of tri-grams
        total_trigrams (int): Total number of tri-grams seen
    """

    def __init__(self, language_name):
        """
        Initialize language model.

        Args:
            language_name (str): Name of the language
        """
        self.language = language_name
        self.trigram_counts = Counter()
        self.total_trigrams = 0

    def train(self, texts):
        """
        Train the model on a collection of texts.

        Args:
            texts (list): List of text strings in the language
        """
        print(f"  Training {self.language} model on {len(texts)} samples...")
        for text in texts:
            trigrams = extract_trigrams(text)
            self.trigram_counts.update(trigrams)
            self.total_trigrams += len(trigrams)

        print(f"  {self.language}: {self.total_trigrams} total tri-grams, "
              f"{len(self.trigram_counts)} unique")

    def get_probability(self, trigram, smoothing=1.0):
        """
        Get probability of a tri-gram under this language model.
        Uses Laplace (add-one) smoothing.

        Args:
            trigram (str): Tri-gram to look up
            smoothing (float): Smoothing parameter (default: 1.0)

        Returns:
            float: Probability of the tri-gram
        """
        count = self.trigram_counts.get(trigram, 0)
        vocab_size = len(self.trigram_counts)
        return (count + smoothing) / (self.total_trigrams + smoothing * vocab_size)


class NaiveBayesLanguageIdentifier:
    """
    Naive Bayes classifier for language identification using character tri-grams.

    Attributes:
        language_models (dict): Dictionary of LanguageModel objects
        language_priors (dict): Prior probabilities for each language
        languages (list): List of supported languages
    """

    def __init__(self):
        """Initialize the classifier."""
        self.language_models = {}
        self.language_priors = {}
        self.languages = []

    def train(self, train_df):
        """
        Train classifier on labeled data.

        Args:
            train_df (DataFrame): DataFrame with 'text' and 'language' columns
        """
        print("\n" + "="*60)
        print("TRAINING NAIVE BAYES CLASSIFIER")
        print("="*60)

        self.languages = sorted(train_df['language'].unique())
        print(f"\nLanguages: {', '.join(self.languages)}")
        print(f"Total training samples: {len(train_df)}\n")

        for language in self.languages:
            texts = train_df[train_df['language'] == language]['text'].tolist()

            model = LanguageModel(language)
            model.train(texts)
            self.language_models[language] = model

            self.language_priors[language] = len(texts) / len(train_df)
            print(f"  Prior P({language}) = {self.language_priors[language]:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

    def _check_short_text(self, text):
        """
        Check whether the input text is too short for reliable prediction.

        Args:
            text (str): Input text

        Returns:
            bool: True if text may be too short for reliable detection
        """
        trigrams = extract_trigrams(text)
        return len(trigrams) < MIN_NGRAMS_FOR_CONFIDENCE

    def predict(self, text):
        """
        Predict the language of a text.

        For very short texts (fewer than 5 tri-grams), a low-confidence
        warning is printed because n-gram statistics are insufficient.

        Args:
            text (str): Input text

        Returns:
            str: Predicted language
        """
        if self._check_short_text(text):
            print(f"  ⚠  Warning: text is very short — prediction may be unreliable.")

        trigrams = extract_trigrams(text)

        best_language = None
        best_log_prob = float('-inf')

        for language, model in self.language_models.items():
            log_prob = np.log(self.language_priors[language])
            for trigram in trigrams:
                prob = model.get_probability(trigram)
                log_prob += np.log(prob)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_language = language

        return best_language

    def predict_with_confidence(self, text):
        """
        Predict language with confidence scores for all languages.

        For very short texts (fewer than 5 tri-grams), the returned dict
        includes an extra key ``'low_confidence'`` set to ``True`` so that
        callers can surface a warning to end-users.

        Args:
            text (str): Input text

        Returns:
            tuple: (predicted_language, confidence_dict)
                - predicted_language (str): Most likely language
                - confidence_dict (dict): Probability for each language,
                  plus optional 'low_confidence' flag
        """
        is_short = self._check_short_text(text)
        trigrams = extract_trigrams(text)

        log_probs = {}

        for language, model in self.language_models.items():
            log_prob = np.log(self.language_priors[language])
            for trigram in trigrams:
                prob = model.get_probability(trigram)
                log_prob += np.log(prob)
            log_probs[language] = log_prob

        # Softmax conversion
        max_log_prob = max(log_probs.values())
        probs = {lang: np.exp(log_prob - max_log_prob)
                 for lang, log_prob in log_probs.items()}

        # Normalize
        total = sum(probs.values())
        probs = {lang: prob / total for lang, prob in probs.items()}

        if is_short:
            probs['low_confidence'] = True

        best_language = max(
            (k for k in probs if k != 'low_confidence'),
            key=lambda k: probs[k]
        )

        return best_language, probs

    def get_supported_languages(self):
        """
        Get list of supported languages.

        Returns:
            list: List of language names
        """
        return self.languages.copy()
