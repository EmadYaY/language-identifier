"""
Programmatic / In-Code Usage Examples
File: in_code_test.py

Demonstrates how to load the trained model and call it programmatically,
without running the interactive or automated test scripts.

This file serves as a quick reference for developers who want to integrate
the language identifier into their own Python code.

Usage:
    python in_code_test.py

Author: Emad
Version: 1.0.0
"""

import pickle
from utils.language_detector_enhanced import EnhancedNaiveBayesLanguageIdentifier


def load_classifier(model_path: str = 'models/language_model.pkl'):
    """
    Load the trained classifier from disk.

    Args:
        model_path (str): Path to the pickled model file.

    Returns:
        classifier: Loaded EnhancedNaiveBayesLanguageIdentifier instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier


def example_simple_predict(classifier):
    """
    Example: simple language prediction (returns language name only).
    """
    print("─" * 50)
    print("Example 1 – simple predict()")
    print("─" * 50)

    sentences = [
        "Eh bien, je vis en Iran, nous sommes en 2025.",
        "درود بر اعضای کمیته پذیرش",
        "Machine learning is changing the world.",
        "Die Sprache ist ein Spiegel der Seele.",
    ]

    for text in sentences:
        language = classifier.predict(text)
        print(f"  '{text[:55]}...' → {language}")

    print()


def example_predict_with_confidence(classifier):
    """
    Example: prediction with confidence scores for all languages.
    """
    print("─" * 50)
    print("Example 2 – predict_with_confidence()")
    print("─" * 50)

    text = "درود بر اعضای کمیته پذیرش"
    language, probabilities = classifier.predict_with_confidence(text)

    # Remove internal flag before display
    low_conf = probabilities.pop('low_confidence', False)

    print(f"  Text:      '{text}'")
    print(f"  Predicted: {language}")
    print(f"  Score:     {probabilities[language]:.2%}")
    if low_conf:
        print("  ⚠ Warning: short text — confidence may be low.")
    print(f"\n  All probabilities:")
    for lang, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {lang:10s}  {prob:6.2%}  {bar}")

    print()


def example_supported_languages(classifier):
    """
    Example: list all languages the classifier was trained on.
    """
    print("─" * 50)
    print("Example 3 – get_supported_languages()")
    print("─" * 50)

    languages = classifier.get_supported_languages()
    print(f"  Supported languages ({len(languages)}): {', '.join(languages)}")
    print()


def main():
    print("=" * 50)
    print("IN-CODE USAGE EXAMPLES  v1.0.0")
    print("=" * 50 + "\n")

    classifier = load_classifier()

    example_simple_predict(classifier)
    example_predict_with_confidence(classifier)
    example_supported_languages(classifier)

    print("=" * 50)
    print("All examples completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
