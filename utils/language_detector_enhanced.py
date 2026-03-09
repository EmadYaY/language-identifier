"""
Enhanced Language Detection with Bigrams + Trigrams
File: utils/language_detector_enhanced.py

This version uses BOTH bigrams and trigrams for better accuracy on short texts
and on closely related languages such as Italian vs. Spanish.

Author: Emad
Version: 1.0.0
"""

import numpy as np
from collections import Counter

# Minimum number of combined n-grams required for confident prediction
MIN_NGRAMS_FOR_CONFIDENCE = 8


def extract_ngrams(text, n=3):
    """
    Extract character n-grams from text.

    Args:
        text (str): Input text
        n (int): Size of n-grams

    Returns:
        list: List of n-grams
    """
    text = text.lower()
    text = '_' + text + '_'
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    return ngrams


def extract_bigrams(text):
    """Extract bigrams (2-character sequences)."""
    return extract_ngrams(text, n=2)


def extract_trigrams(text):
    """Extract trigrams (3-character sequences)."""
    return extract_ngrams(text, n=3)


def extract_combined_features(text):
    """
    Extract BOTH bigrams and trigrams for better accuracy.

    Bigrams are prefixed with '2_' and trigrams with '3_' to avoid
    collisions between n-gram levels in the shared vocabulary.

    Args:
        text (str): Input text

    Returns:
        list: Combined list of tagged bigrams and trigrams
    """
    bigrams = extract_bigrams(text)
    trigrams = extract_trigrams(text)

    bigrams_tagged = [f"2_{bg}" for bg in bigrams]
    trigrams_tagged = [f"3_{tg}" for tg in trigrams]

    return bigrams_tagged + trigrams_tagged


class EnhancedLanguageModel:
    """
    Enhanced language model using both bigrams and trigrams.

    Attributes:
        language (str): Name of the language
        ngram_counts (Counter): Frequency counts of combined n-grams
        total_ngrams (int): Total number of n-grams seen
    """

    def __init__(self, language_name):
        self.language = language_name
        self.ngram_counts = Counter()
        self.total_ngrams = 0

    def train(self, texts):
        """
        Train the model on a collection of texts.

        Args:
            texts (list): List of text strings in the language
        """
        print(f"  Training {self.language} model on {len(texts)} samples...")
        for text in texts:
            ngrams = extract_combined_features(text)
            self.ngram_counts.update(ngrams)
            self.total_ngrams += len(ngrams)

        print(f"  {self.language}: {self.total_ngrams} total n-grams, "
              f"{len(self.ngram_counts)} unique")

    def get_probability(self, ngram, smoothing=1.0):
        """
        Get probability of an n-gram with Laplace smoothing.

        Args:
            ngram (str): Tagged n-gram (e.g. '3_hel', '2_he')
            smoothing (float): Smoothing parameter (default: 1.0)

        Returns:
            float: Smoothed probability of the n-gram
        """
        count = self.ngram_counts.get(ngram, 0)
        vocab_size = len(self.ngram_counts)
        return (count + smoothing) / (self.total_ngrams + smoothing * vocab_size)


class EnhancedNaiveBayesLanguageIdentifier:
    """
    Enhanced Naive Bayes classifier using bigrams + trigrams.

    This classifier outperforms the trigram-only baseline especially on:
    - Short texts (single words or short phrases)
    - Closely related languages (e.g. Italian vs. Spanish)

    Attributes:
        language_models (dict): Dictionary of EnhancedLanguageModel objects
        language_priors (dict): Prior probabilities for each language
        languages (list): List of supported languages
    """

    def __init__(self):
        """Initialize the enhanced classifier."""
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
        print("TRAINING ENHANCED NAIVE BAYES CLASSIFIER")
        print("Using BOTH bigrams and trigrams")
        print("="*60)

        self.languages = sorted(train_df['language'].unique())
        print(f"\nLanguages: {', '.join(self.languages)}")
        print(f"Total training samples: {len(train_df)}\n")

        for language in self.languages:
            texts = train_df[train_df['language'] == language]['text'].tolist()

            model = EnhancedLanguageModel(language)
            model.train(texts)
            self.language_models[language] = model

            self.language_priors[language] = len(texts) / len(train_df)
            print(f"  Prior P({language}) = {self.language_priors[language]:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

    def _check_short_text(self, text):
        """
        Check whether the input text may be too short for reliable prediction.

        Args:
            text (str): Input text

        Returns:
            bool: True if feature count is below the minimum threshold
        """
        ngrams = extract_combined_features(text)
        return len(ngrams) < MIN_NGRAMS_FOR_CONFIDENCE

    def predict(self, text):
        """
        Predict the language of a text.

        For very short texts, a low-confidence warning is printed.

        Args:
            text (str): Input text

        Returns:
            str: Predicted language
        """
        if self._check_short_text(text):
            print(f"  ⚠  Warning: text is very short — prediction may be unreliable.")

        ngrams = extract_combined_features(text)

        best_language = None
        best_log_prob = float('-inf')

        for language, model in self.language_models.items():
            log_prob = np.log(self.language_priors[language])
            for ngram in ngrams:
                prob = model.get_probability(ngram)
                log_prob += np.log(prob)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_language = language

        return best_language

    def predict_with_confidence(self, text):
        """
        Predict language with confidence scores for all languages.

        For short texts, the returned dict includes a ``'low_confidence'``
        key set to ``True`` so callers can display a suitable warning.

        Args:
            text (str): Input text

        Returns:
            tuple: (predicted_language, confidence_dict)
                - predicted_language (str): Most likely language
                - confidence_dict (dict): Probability per language,
                  plus optional 'low_confidence' flag
        """
        is_short = self._check_short_text(text)
        ngrams = extract_combined_features(text)

        log_probs = {}

        for language, model in self.language_models.items():
            log_prob = np.log(self.language_priors[language])
            for ngram in ngrams:
                prob = model.get_probability(ngram)
                log_prob += np.log(prob)
            log_probs[language] = log_prob

        # Softmax
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


# Backward-compatibility aliases so existing code that imports from this
# module continues to work without modification.
NaiveBayesLanguageIdentifier = EnhancedNaiveBayesLanguageIdentifier
LanguageModel = EnhancedLanguageModel
