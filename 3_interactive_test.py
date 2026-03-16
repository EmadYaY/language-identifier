"""
Language Identification Model - Interactive Testing Script
File: 3_interactive_test.py

Allows users to input arbitrary text and receive:
  • predicted language
  • confidence score derived from posterior probabilities
  • low-confidence warning for very short inputs
  • optional probability bar chart for all languages

Usage:
    python 3_interactive_test.py                   # interactive mode
    python 3_interactive_test.py "text1" "text2"   # batch mode

Author: Emad
Version: 1.0.0
"""

import pickle
import sys

from utils.language_detector_enhanced import EnhancedNaiveBayesLanguageIdentifier


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(model_path='models/language_model.pkl'):
    """
    Load the trained model from disk.

    Args:
        model_path (str): Path to the pickled model file

    Returns:
        classifier: Loaded classifier, or None on error
    """
    try:
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        return classifier
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python 1_train_model_v3.py")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def display_prediction(text, classifier, show_all_probs=True):
    """
    Display prediction results for a given text.

    Args:
        text (str): Input text
        classifier: Trained classifier
        show_all_probs (bool): Whether to show the full probability breakdown
    """
    predicted_lang, probs = classifier.predict_with_confidence(text)
    low_conf = probs.pop('low_confidence', False)

    print("\n" + "="*70)
    print(f"📝 Input text: {text}")
    print("-"*70)
    print(f"🌍 Detected Language: {predicted_lang.upper()}")
    print(f"📊 Confidence:        {probs[predicted_lang]:.2%}")

    if low_conf:
        print("⚠  Warning: text is very short — result may be unreliable.")

    if show_all_probs:
        print(f"\n📈 All Language Probabilities:")
        print("-"*70)

        for lang, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 50)
            marker = " ⭐" if lang == predicted_lang else ""
            prefix = "  ➤" if lang == predicted_lang else "   "
            print(f"{prefix} {lang:10s} {prob:6.2%}  {bar}{marker}")

    print("="*70)


def print_examples():
    """Print example sentences for each language."""
    print("\n" + "="*70)
    print("📚 EXAMPLE SENTENCES")
    print("="*70)

    examples = {
        'English': [
            "Hello, how are you?",
            "Machine learning is amazing.",
            "The weather is beautiful today.",
        ],
        'Spanish': [
            "Hola, ¿cómo estás?",
            "El aprendizaje automático es increíble.",
            "El clima está hermoso hoy.",
        ],
        'French': [
            "Bonjour, comment allez-vous?",
            "L'apprentissage automatique est incroyable.",
            "Le temps est magnifique aujourd'hui.",
        ],
        'German': [
            "Hallo, wie geht es dir?",
            "Maschinelles Lernen ist erstaunlich.",
            "Das Wetter ist heute wunderschön.",
        ],
        'Italian': [
            "Ciao, come stai?",
            "L'apprendimento automatico è fantastico.",
            "Il tempo è bellissimo oggi.",
        ],
        'Persian': [
            "سلام، حالت چطوره؟",
            "یادگیری ماشین شگفت‌انگیز است.",
            "هوا امروز زیباست.",
        ],
    }

    for lang, sentences in examples.items():
        print(f"\n{lang}:")
        for sentence in sentences:
            print(f"  • {sentence}")

    print("="*70 + "\n")


# ─────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────

def interactive_mode(classifier):
    """
    Interactive REPL for continuous language detection.

    Commands:
        quit / exit / q  — stop
        help             — show example sentences
        short            — toggle detailed probability output

    Args:
        classifier: Trained classifier
    """
    print("\n" + "="*70)
    print("🔤 INTERACTIVE LANGUAGE DETECTION")
    print("="*70)
    print("\nEnter text in any language to detect it.")
    print("Supported languages:", ", ".join(classifier.languages))
    print("\nCommands:")
    print("  • 'quit' / 'exit' / 'q' — stop")
    print("  • 'help'                 — show example sentences")
    print("  • 'short'                — toggle detailed probability output")
    print("="*70 + "\n")

    show_details = True

    while True:
        try:
            user_input = input("➤ Enter text: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            elif user_input.lower() == 'help':
                print_examples()
                continue

            elif user_input.lower() == 'short':
                show_details = not show_details
                mode = "detailed" if show_details else "simple"
                print(f"\n✓ Switched to {mode} mode\n")
                continue

            elif not user_input:
                print("⚠  Please enter some text.\n")
                continue

            display_prediction(user_input, classifier, show_all_probs=show_details)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (interrupted by user)")
            break

        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def batch_mode(classifier, texts):
    """
    Batch mode: process a list of texts and print a summary table.

    Args:
        classifier: Trained classifier
        texts (list): List of strings to classify

    Returns:
        list[dict]: List of result dicts with 'text' and 'language' keys
    """
    print("\n" + "="*70)
    print("📦 BATCH PROCESSING MODE")
    print("="*70)

    results = []

    for i, text in enumerate(texts, 1):
        pred_lang, probs = classifier.predict_with_confidence(text)
        low_conf = probs.pop('low_confidence', False)
        warning = " ⚠" if low_conf else ""

        results.append({'index': i, 'text': text, 'language': pred_lang})
        print(f"\n{i}. Text: {text[:65]}{'...' if len(text) > 65 else ''}")
        print(f"   Language: {pred_lang}{warning}")

    print("\n" + "="*70)
    print(f"✓ Processed {len(texts)} text(s)")
    print("="*70)

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """Entry point: selects interactive or batch mode based on CLI args."""

    print("="*70)
    print("🌍 LANGUAGE IDENTIFICATION - INTERACTIVE MODE  v1.0.0")
    print("="*70)
    print("\nLoading model...")

    classifier = load_model()
    if classifier is None:
        return

    print(f"✓ Model loaded successfully!")
    print(f"✓ Supported languages: {', '.join(classifier.languages)}")

    if len(sys.argv) > 1:
        batch_mode(classifier, sys.argv[1:])
    else:
        interactive_mode(classifier)


if __name__ == "__main__":
    main()
