# Language Identification System

**Character N-gram Naive Bayes — v1.0.0**

A lightweight, fully self-implemented language identification system based on
character-level n-grams (bigrams + trigrams) and a Multinomial Naive Bayes
classifier. No external language-detection libraries are used; every step from
preprocessing to inference is implemented from scratch.

---

## Supported Languages

| Language | Script       |
|----------|--------------|
| English  | Latin        |
| German   | Latin        |
| French   | Latin        |
| Spanish  | Latin        |
| Italian  | Latin        |
| Persian  | Perso-Arabic |

The system is easily extensible: add labelled samples in `1_train_model_v3.py`
and retrain.

---

## Project Structure

```
language_identifier_project/
│
├── utils/
│   ├── language_detector.py          # Trigram-only classifier (baseline)
│   └── language_detector_enhanced.py # Bigram + trigram classifier (default)
│
├── models/
│   └── language_model.pkl            # Serialised trained model
│
├── test_data/
│   ├── test_samples.csv              # Evaluation dataset
│   └── test_samples_predictions.csv  # Model output (generated)
│
├── Old Versions/                     # Archived development iterations
│
├── 1_train_model_v3.py               # Training script (enhanced classifier)
├── 2_test_model.py                   # Automated evaluation + confusion matrix
├── 3_interactive_test.py             # Interactive / batch inference interface
├── in_code_test.py                   # Programmatic usage examples
└── README.md
```

---

## Quick Start

### Requirements

```
Python >= 3.8
numpy
pandas
scikit-learn
```

```bash
pip install numpy pandas scikit-learn
```

### 1 — Train the model

```bash
python 1_train_model_v3.py
```

Expected output summary:

```
🎯 Overall Accuracy: ~88–95%

📊 Per-Language Accuracy:
  🟢 english  : 90%+
  🟢 french   : 90%+
  🟢 german   : 90%+
  🟢 persian  : 100%
  🟡 italian  : 80%+
  🟢 spanish  : 90%+
```

### 2 — Run automated tests

```bash
python 2_test_model.py
```

Produces four test sections:

- **Test 1** — predefined sentences (18 samples)
- **Test 2** — CSV evaluation with saved predictions
- **Test 3** — detailed per-language confidence bars
- **Test 4** — confusion matrix

### 3 — Interactive / batch inference

```bash
# Interactive REPL
python 3_interactive_test.py

# Batch mode (pass texts as arguments)
python 3_interactive_test.py "Bonjour le monde" "Ciao come stai" "Hello world"
```

---

## Programmatic Usage

```python
import pickle
from utils.language_detector_enhanced import EnhancedNaiveBayesLanguageIdentifier

with open('models/language_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Simple prediction
language = classifier.predict("This is an English sentence.")
print(language)  # → english

# Prediction with confidence scores
language, probs = classifier.predict_with_confidence("Bonjour le monde")
low_conf = probs.pop('low_confidence', False)   # True for very short inputs
print(language, f"{probs[language]:.2%}")       # → french  99.80%

# List supported languages
print(classifier.get_supported_languages())
# → ['english', 'french', 'german', 'italian', 'persian', 'spanish']
```

---

## Technical Details

### Feature Extraction

The **enhanced classifier** (used by default) combines two n-gram levels:

| Level   | Tag prefix | Example input `hello` → features         |
|---------|------------|------------------------------------------|
| Bigram  | `2_`       | `2__h`, `2_he`, `2_el`, `2_ll`, `2_lo`, `2_o_` |
| Trigram | `3_`       | `3__he`, `3_hel`, `3_ell`, `3_llo`, `3_lo_`   |

Prefixes prevent collisions between the two levels in the shared vocabulary.
The baseline `language_detector.py` uses trigrams only and is kept for
comparison.

### Classification

- **Model:** Multinomial Naive Bayes
- **Smoothing:** Laplace (add-one), parameter α = 1.0
- **Numerics:** log-probability accumulation to prevent underflow
- **Normalisation:** softmax over log-posteriors for calibrated confidence scores
- **Priors:** estimated from training class frequencies

### Short-text Handling

Inputs with fewer combined n-grams than a configurable threshold
(`MIN_NGRAMS_FOR_CONFIDENCE`) trigger a `low_confidence` flag in
`predict_with_confidence()` and a console warning in `predict()`.
This makes unreliable single-word predictions transparent to callers.

### Training Split

80 % training / 20 % test, stratified by language, `random_state=42`.

---

## Performance

On the built-in test set the enhanced model consistently achieves:

| Metric          | Value     |
|-----------------|-----------|
| Overall accuracy | ~88–95 % |
| Persian          | ~100 %   |
| German / French  | ~90 %    |
| English          | ~85–90 % |
| Italian / Spanish| ~80–90 % |

Italian and Spanish share a large portion of their Latin-root vocabulary, which
is the primary source of confusion. Adding more diverse training samples
(news articles, literature) reliably pushes accuracy above 95 % for both.

---

## Extending the System

1. Add labelled samples for the new language in `create_complete_dataset()`
   inside `1_train_model_v3.py`.
2. Retrain:
   ```bash
   python 1_train_model_v3.py
   ```
3. Re-run evaluation:
   ```bash
   python 2_test_model.py
   ```

---

## Reference

Cavnar, W. B., & Trenkle, J. M. (1994).
*N-Gram-Based Text Categorization.*
Proceedings of the Third Annual Symposium on Document Analysis and Information
Retrieval (SDAIR-94).

---

## Author

**Emad** — v1.0.0
