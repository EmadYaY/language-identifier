"""
Language Identification Model - Automated Testing Script
File: 2_test_model.py

Runs automated tests on the trained language identification model.
Includes predefined sentence tests, CSV-based evaluation, detailed
confidence analysis, and a confusion matrix summary.

Usage:
    python 2_test_model.py

Author: Emad
Version: 1.0.0
"""

import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import our custom classes (enhanced variant used by the saved model)
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
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Supported languages: {', '.join(classifier.languages)}")
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
# Test 1 – predefined sentences
# ─────────────────────────────────────────────

def test_predefined_sentences(classifier):
    """
    Evaluate the classifier on a fixed set of hand-picked sentences.

    Args:
        classifier: Trained classifier
    """
    print("\n" + "="*70)
    print("TEST 1: PREDEFINED SENTENCES")
    print("="*70)

    test_cases = [
        ("english", "Hello, how are you today?"),
        ("english", "Machine learning is fascinating."),
        ("english", "The weather is beautiful today."),

        ("spanish", "¿Cómo estás hoy?"),
        ("spanish", "El aprendizaje automático es fascinante."),
        ("spanish", "El clima está hermoso hoy."),

        ("french",  "Comment allez-vous aujourd'hui?"),
        ("french",  "L'apprentissage automatique est fascinant."),
        ("french",  "Le temps est magnifique aujourd'hui."),

        ("german",  "Wie geht es dir heute?"),
        ("german",  "Maschinelles Lernen ist faszinierend."),
        ("german",  "Das Wetter ist heute wunderschön."),

        ("italian", "Come stai oggi?"),
        ("italian", "L'apprendimento automatico è affascinante."),
        ("italian", "Il tempo è bellissimo oggi."),

        ("persian", "امروز حال شما چطور است؟"),
        ("persian", "یادگیری ماشین جذاب است."),
        ("persian", "هوا امروز زیبا است."),
    ]

    correct = 0
    total = len(test_cases)
    print(f"\nTesting {total} sentences...\n")

    for expected_lang, text in test_cases:
        predicted_lang, probs = classifier.predict_with_confidence(text)
        low_conf = probs.pop('low_confidence', False)
        is_correct = predicted_lang == expected_lang

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        confidence = probs[predicted_lang]
        warning = " ⚠" if low_conf else ""

        print(f"{status} Expected: {expected_lang:10s} | "
              f"Predicted: {predicted_lang:10s} ({confidence:.1%}){warning}")
        print(f"   Text: {text[:65]}")

    accuracy = (correct / total) * 100
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("="*70)

    return correct, total


# ─────────────────────────────────────────────
# Test 2 – CSV evaluation
# ─────────────────────────────────────────────

def test_from_csv(classifier, csv_path='test_data/test_samples.csv'):
    """
    Evaluate the classifier on samples from a CSV file.

    The CSV must contain a 'text' column. An optional 'language' column
    enables accuracy reporting. Predictions are saved to a new CSV.

    Args:
        classifier: Trained classifier
        csv_path (str): Path to the input CSV file
    """
    print("\n" + "="*70)
    print("TEST 2: FROM CSV FILE")
    print("="*70)

    if not Path(csv_path).exists():
        print(f"\nCreating sample CSV file: {csv_path}")
        create_sample_csv(csv_path)

    try:
        df = pd.read_csv(csv_path)

        if 'text' not in df.columns:
            print("✗ Error: CSV must have a 'text' column")
            return

        has_labels = 'language' in df.columns

        print(f"\nFile: {csv_path}")
        print(f"Total samples: {len(df)}\n")

        predictions = []
        correct = 0

        for idx, row in df.iterrows():
            text = row['text']
            pred_lang, probs = classifier.predict_with_confidence(text)
            low_conf = probs.pop('low_confidence', False)
            predictions.append(pred_lang)

            warning = " ⚠" if low_conf else ""

            if has_labels:
                true_lang = row['language']
                is_correct = pred_lang == true_lang
                if is_correct:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"{idx+1:2d}. {status} True: {true_lang:10s} | "
                      f"Pred: {pred_lang:10s}{warning} | {text[:45]}...")
            else:
                print(f"{idx+1:2d}. Predicted: {pred_lang:10s}{warning} | {text[:55]}...")

        if has_labels:
            accuracy = (correct / len(df)) * 100
            print(f"\n{'='*70}")
            print(f"Accuracy: {correct}/{len(df)} = {accuracy:.1f}%")
            print("="*70)

        # Save predictions
        df['predicted_language'] = predictions
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Predictions saved to: {output_path}")

    except FileNotFoundError:
        print(f"✗ Error: File not found '{csv_path}'")
    except Exception as e:
        print(f"✗ Error: {e}")


def create_sample_csv(csv_path):
    """
    Create a sample CSV file for testing.

    Args:
        csv_path (str): Destination path
    """
    sample_data = {
        'text': [
            "Artificial intelligence is changing the world.",
            "Machine learning algorithms can solve complex problems.",
            "Python is a great programming language.",

            "La inteligencia artificial está cambiando el mundo.",
            "Los algoritmos de aprendizaje automático pueden resolver problemas complejos.",
            "Python es un gran lenguaje de programación.",

            "L'intelligence artificielle change le monde.",
            "Les algorithmes d'apprentissage automatique peuvent résoudre des problèmes complexes.",
            "Python est un excellent langage de programmation.",

            "Künstliche Intelligenz verändert die Welt.",
            "Maschinelle Lernalgorithmen können komplexe Probleme lösen.",
            "Python ist eine großartige Programmiersprache.",

            "L'intelligenza artificiale sta cambiando il mondo.",
            "Gli algoritmi di apprendimento automatico possono risolvere problemi complessi.",
            "Python è un ottimo linguaggio di programmazione.",

            "هوش مصنوعی در حال تغییر جهان است.",
            "الگوریتم‌های یادگیری ماشین می‌توانند مسائل پیچیده را حل کنند.",
            "پایتون یک زبان برنامه‌نویسی عالی است.",
        ],
        'language': [
            'english', 'english', 'english',
            'spanish', 'spanish', 'spanish',
            'french',  'french',  'french',
            'german',  'german',  'german',
            'italian', 'italian', 'italian',
            'persian', 'persian', 'persian',
        ]
    }

    df = pd.DataFrame(sample_data)
    Path(csv_path).parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Sample CSV created: {csv_path}")


# ─────────────────────────────────────────────
# Test 3 – detailed confidence analysis
# ─────────────────────────────────────────────

def test_detailed_analysis(classifier):
    """
    Show per-language probability bars for a small set of sentences.

    Args:
        classifier: Trained classifier
    """
    print("\n" + "="*70)
    print("TEST 3: DETAILED CONFIDENCE ANALYSIS")
    print("="*70)

    test_sentences = {
        'English': "This is a test sentence in English about artificial intelligence.",
        'Spanish': "Esta es una oración de prueba en español sobre inteligencia artificial.",
        'Persian': "این یک جمله آزمایشی به زبان فارسی درباره هوش مصنوعی است.",
        'Italian': "Questa è una frase di prova in italiano sull'intelligenza artificiale.",
        'German':  "Dies ist ein Testsatz auf Deutsch über künstliche Intelligenz.",
    }

    for true_lang, text in test_sentences.items():
        pred_lang, probs = classifier.predict_with_confidence(text)
        low_conf = probs.pop('low_confidence', False)
        warning = "  ⚠ low-confidence" if low_conf else ""

        print(f"\n{'-'*70}")
        print(f"Text: {text}")
        print(f"True language:  {true_lang}")
        print(f"Predicted:      {pred_lang.upper()} ({probs[pred_lang]:.2%}){warning}")
        print(f"\nAll probabilities:")
        for lang, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            marker = " ⭐" if lang == pred_lang else ""
            print(f"  {lang:10s} {prob:6.2%}  {bar}{marker}")


# ─────────────────────────────────────────────
# Test 4 – confusion matrix
# ─────────────────────────────────────────────

def test_confusion_matrix(classifier):
    """
    Run bulk predictions over all predefined sentences and print a
    confusion matrix to highlight inter-language confusions.

    Args:
        classifier: Trained classifier
    """
    print("\n" + "="*70)
    print("TEST 4: CONFUSION MATRIX ON PREDEFINED SET")
    print("="*70)

    test_cases = [
        # English
        ("english", "Hello, how are you today?"),
        ("english", "Machine learning is fascinating."),
        ("english", "The weather is beautiful today."),
        ("english", "Software engineering is creative work."),
        ("english", "Natural language processing is a growing field."),
        # Spanish
        ("spanish", "¿Cómo estás hoy?"),
        ("spanish", "El aprendizaje automático es fascinante."),
        ("spanish", "Los algoritmos de machine learning son poderosos."),
        ("spanish", "La tecnología avanza muy rápido."),
        ("spanish", "El sol sale por el este cada mañana."),
        # French
        ("french",  "Comment allez-vous aujourd'hui?"),
        ("french",  "L'apprentissage automatique est fascinant."),
        ("french",  "La science des données est en plein essor."),
        ("french",  "Le temps est magnifique ce matin."),
        ("french",  "Les algorithmes de machine learning sont puissants."),
        # German
        ("german",  "Wie geht es dir heute?"),
        ("german",  "Maschinelles Lernen ist faszinierend."),
        ("german",  "Das Wetter ist heute wunderschön."),
        ("german",  "Technologie schreitet schnell voran."),
        ("german",  "Datenwissenschaft ist sehr interessant."),
        # Italian
        ("italian", "Come stai oggi?"),
        ("italian", "L'apprendimento automatico è affascinante."),
        ("italian", "Il tempo è bellissimo oggi."),
        ("italian", "La tecnologia avanza molto rapidamente."),
        ("italian", "La scienza dei dati è in forte crescita."),
        # Persian
        ("persian", "امروز حال شما چطور است؟"),
        ("persian", "یادگیری ماشین جذاب است."),
        ("persian", "هوا امروز زیبا است."),
        ("persian", "تکنولوژی به سرعت پیشرفت می‌کند."),
        ("persian", "علم داده در حال رشد است."),
    ]

    languages = sorted(set(t for t, _ in test_cases))
    confusion = {t: defaultdict(int) for t in languages}
    correct = 0

    for true_lang, text in test_cases:
        pred_lang, probs = classifier.predict_with_confidence(text)
        probs.pop('low_confidence', None)
        confusion[true_lang][pred_lang] += 1
        if pred_lang == true_lang:
            correct += 1

    # Print matrix
    col_w = 10
    print(f"\n  {'':12s}" + "".join(f"{lang:>{col_w}}" for lang in languages))
    print("  " + "-" * (12 + col_w * len(languages)))

    for true_lang in languages:
        row = f"  {true_lang:<12s}"
        for pred_lang in languages:
            count = confusion[true_lang].get(pred_lang, 0)
            cell = f"[{count}]" if true_lang == pred_lang else f" {count} "
            row += f"{cell:>{col_w}}"
        print(row)

    accuracy = correct / len(test_cases)
    print(f"\n  Overall: {correct}/{len(test_cases)} = {accuracy:.1%}")
    print("="*70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """Main testing function."""

    print("="*70)
    print("LANGUAGE IDENTIFICATION MODEL - AUTOMATED TESTING  v1.0.0")
    print("="*70)

    classifier = load_model()
    if classifier is None:
        return

    test_predefined_sentences(classifier)
    test_from_csv(classifier)
    test_detailed_analysis(classifier)
    test_confusion_matrix(classifier)

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)
    print("\nNext step:")
    print("  Run '3_interactive_test.py' for manual testing")


if __name__ == "__main__":
    main()
