"""
Language Identification System - Training Script
File: 1_train_model_v3.py

Trains a character n-gram Naive Bayes language identifier on a built-in
labelled dataset (no external data dependencies).

The classifier used here is the *enhanced* variant that combines bigrams
and trigrams, which improves accuracy on short texts and on closely related
Romance languages (Italian / Spanish).

Usage:
    python 1_train_model_v3.py

Author: Emad
Version: 1.0.0
Reference: Cavnar, W. B., & Trenkle, J. M. (1994). N-Gram-Based Text Categorization.
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Use the enhanced classifier (bigrams + trigrams) for better accuracy
from utils.language_detector_enhanced import EnhancedNaiveBayesLanguageIdentifier


def create_complete_dataset():
    """
    Create a labelled dataset with 50 diverse samples per language.
    Total: 50 × 6 = 300 samples.

    Returns:
        DataFrame with 'text' and 'language' columns
    """
    print("\n✓ Creating local dataset (50 samples × 6 languages = 300 total)")

    dataset = {
        'english': [
            "Hello, how are you?", "Good morning!", "Thank you very much.",
            "Machine learning is fascinating.", "Python is a great language.",
            "I love programming.", "The weather is nice today.",
            "Artificial intelligence is amazing.", "Data science is interesting.",
            "Natural language processing helps computers.", "Deep learning uses neural networks.",
            "I am learning English now.", "Can you help me please?",
            "The quick brown fox jumps.", "Software engineering requires creativity.",
            "Cybersecurity is very important.", "Cloud computing is everywhere.",
            "The internet connects people.", "Technology advances rapidly.",
            "Music makes me happy.", "I like reading books.",
            "Coffee is my favorite drink.", "Exercise is good for health.",
            "Travel broadens the mind.", "Education opens doors.",
            "The sun rises in east.", "Water is essential for life.",
            "Birds fly in the sky.", "Flowers bloom in spring.",
            "I went shopping yesterday.", "She is reading a book.",
            "They will arrive tomorrow.", "He works in a company.",
            "Children play in park.", "Students study at library.",
            "The car is very fast.", "This house is old.",
            "My phone battery is low.", "The movie was good.",
            "I need to buy groceries.", "Let's go to restaurant.",
            "The train leaves at noon.", "Can you speak slowly?",
            "I would like some tea.", "It's raining outside now.",
            "The food tastes delicious.", "I'm tired after work.",
            "We should leave soon.", "The test was difficult.",
            "I forgot my password.",
        ],

        'spanish': [
            "Hola, ¿cómo estás?", "Buenos días.", "Muchas gracias.",
            "El aprendizaje automático es fascinante.", "Python es un gran lenguaje.",
            "Me encanta programar.", "El clima está agradable hoy.",
            "La inteligencia artificial es increíble.", "La ciencia de datos es interesante.",
            "El procesamiento del lenguaje natural ayuda.", "El aprendizaje profundo usa redes neuronales.",
            "Estoy aprendiendo español ahora.", "¿Puedes ayudarme por favor?",
            "El rápido zorro marrón salta.", "La ingeniería de software requiere creatividad.",
            "La ciberseguridad es muy importante.", "La computación en nube está en todas partes.",
            "Internet conecta a las personas.", "La tecnología avanza rápidamente.",
            "La música me hace feliz.", "Me gusta leer libros.",
            "El café es mi bebida favorita.", "El ejercicio es bueno para la salud.",
            "Viajar amplía la mente.", "La educación abre puertas.",
            "El sol sale por el este.", "El agua es esencial para la vida.",
            "Los pájaros vuelan en el cielo.", "Las flores florecen en primavera.",
            "Fui de compras ayer.", "Ella está leyendo un libro.",
            "Llegarán mañana.", "Él trabaja en una empresa.",
            "Los niños juegan en el parque.", "Los estudiantes estudian en la biblioteca.",
            "El coche es muy rápido.", "Esta casa es vieja.",
            "La batería de mi teléfono está baja.", "La película fue buena.",
            "Necesito comprar comestibles.", "Vamos al restaurante.",
            "El tren sale al mediodía.", "¿Puedes hablar despacio?",
            "Me gustaría un poco de té.", "Está lloviendo afuera ahora.",
            "La comida sabe deliciosa.", "Estoy cansado después del trabajo.",
            "Deberíamos irnos pronto.", "El examen fue difícil.",
            "Olvidé mi contraseña.",
        ],

        'french': [
            "Bonjour, comment allez-vous?", "Bonne journée!", "Merci beaucoup.",
            "L'apprentissage automatique est fascinant.", "Python est un excellent langage.",
            "J'adore programmer.", "Le temps est agréable aujourd'hui.",
            "L'intelligence artificielle est incroyable.", "La science des données est intéressante.",
            "Le traitement du langage naturel aide.", "L'apprentissage profond utilise des réseaux.",
            "J'apprends le français maintenant.", "Pouvez-vous m'aider s'il vous plaît?",
            "Le rapide renard brun saute.", "Le génie logiciel nécessite de la créativité.",
            "La cybersécurité est très importante.", "Le cloud computing est partout.",
            "Internet connecte les gens.", "La technologie progresse rapidement.",
            "La musique me rend heureux.", "J'aime lire des livres.",
            "Le café est ma boisson préférée.", "L'exercice est bon pour la santé.",
            "Voyager élargit l'esprit.", "L'éducation ouvre des portes.",
            "Le soleil se lève à l'est.", "L'eau est essentielle à la vie.",
            "Les oiseaux volent dans le ciel.", "Les fleurs fleurissent au printemps.",
            "Je suis allé faire du shopping hier.", "Elle lit un livre.",
            "Ils arriveront demain.", "Il travaille dans une entreprise.",
            "Les enfants jouent dans le parc.", "Les étudiants étudient à la bibliothèque.",
            "La voiture est très rapide.", "Cette maison est vieille.",
            "Ma batterie de téléphone est faible.", "Le film était bon.",
            "Je dois acheter des courses.", "Allons au restaurant.",
            "Le train part à midi.", "Pouvez-vous parler lentement?",
            "Je voudrais du thé.", "Il pleut dehors maintenant.",
            "La nourriture a bon goût.", "Je suis fatigué après le travail.",
            "Nous devrions partir bientôt.", "L'examen était difficile.",
            "J'ai oublié mon mot de passe.",
        ],

        'german': [
            "Hallo, wie geht es dir?", "Guten Morgen!", "Vielen Dank.",
            "Maschinelles Lernen ist faszinierend.", "Python ist eine großartige Sprache.",
            "Ich liebe Programmieren.", "Das Wetter ist heute schön.",
            "Künstliche Intelligenz ist erstaunlich.", "Datenwissenschaft ist interessant.",
            "Natürliche Sprachverarbeitung hilft.", "Deep Learning verwendet neuronale Netze.",
            "Ich lerne jetzt Deutsch.", "Kannst du mir bitte helfen?",
            "Der schnelle braune Fuchs springt.", "Software-Engineering erfordert Kreativität.",
            "Cybersicherheit ist sehr wichtig.", "Cloud Computing ist überall.",
            "Das Internet verbindet Menschen.", "Technologie schreitet schnell voran.",
            "Musik macht mich glücklich.", "Ich lese gerne Bücher.",
            "Kaffee ist mein Lieblingsgetränk.", "Bewegung ist gut für die Gesundheit.",
            "Reisen erweitert den Geist.", "Bildung öffnet Türen.",
            "Die Sonne geht im Osten auf.", "Wasser ist für das Leben unerlässlich.",
            "Vögel fliegen am Himmel.", "Blumen blühen im Frühling.",
            "Ich bin gestern einkaufen gegangen.", "Sie liest ein Buch.",
            "Sie werden morgen ankommen.", "Er arbeitet in einem Unternehmen.",
            "Kinder spielen im Park.", "Studenten lernen in der Bibliothek.",
            "Das Auto ist sehr schnell.", "Dieses Haus ist alt.",
            "Mein Telefonakku ist schwach.", "Der Film war gut.",
            "Ich muss Lebensmittel kaufen.", "Lass uns zum Restaurant gehen.",
            "Der Zug fährt mittags ab.", "Kannst du langsam sprechen?",
            "Ich hätte gerne etwas Tee.", "Es regnet jetzt draußen.",
            "Das Essen schmeckt köstlich.", "Ich bin nach der Arbeit müde.",
            "Wir sollten bald gehen.", "Der Test war schwierig.",
            "Ich habe mein Passwort vergessen.",
        ],

        'italian': [
            "Ciao, come stai?", "Buongiorno!", "Grazie mille.",
            "L'apprendimento automatico è affascinante.", "Python è un ottimo linguaggio.",
            "Amo programmare.", "Il tempo è bello oggi.",
            "L'intelligenza artificiale è incredibile.", "La scienza dei dati è interessante.",
            "L'elaborazione del linguaggio naturale aiuta.", "Il deep learning usa reti neurali.",
            "Sto imparando l'italiano adesso.", "Puoi aiutarmi per favore?",
            "La veloce volpe marrone salta.", "L'ingegneria del software richiede creatività.",
            "La cybersicurezza è molto importante.", "Il cloud computing è ovunque.",
            "Internet connette le persone.", "La tecnologia avanza rapidamente.",
            "La musica mi rende felice.", "Mi piace leggere libri.",
            "Il caffè è la mia bevanda preferita.", "L'esercizio fa bene alla salute.",
            "Viaggiare allarga la mente.", "L'istruzione apre porte.",
            "Il sole sorge a est.", "L'acqua è essenziale per la vita.",
            "Gli uccelli volano nel cielo.", "I fiori sbocciano in primavera.",
            "Sono andato a fare shopping ieri.", "Lei sta leggendo un libro.",
            "Arriveranno domani.", "Lui lavora in un'azienda.",
            "I bambini giocano nel parco.", "Gli studenti studiano in biblioteca.",
            "L'auto è molto veloce.", "Questa casa è vecchia.",
            "La batteria del mio telefono è scarica.", "Il film era buono.",
            "Devo comprare la spesa.", "Andiamo al ristorante.",
            "Il treno parte a mezzogiorno.", "Puoi parlare lentamente?",
            "Vorrei del tè.", "Sta piovendo fuori adesso.",
            "Il cibo ha un buon sapore.", "Sono stanco dopo il lavoro.",
            "Dovremmo partire presto.", "L'esame era difficile.",
            "Ho dimenticato la mia password.",
        ],

        'persian': [
            "سلام، حالت چطوره؟", "صبح بخیر!", "خیلی ممنون.",
            "یادگیری ماشین جذابه.", "پایتون زبان عالیه.",
            "عاشق برنامه‌نویسی هستم.", "امروز هوا خوبه.",
            "هوش مصنوعی شگفت‌انگیزه.", "علم داده جالبه.",
            "پردازش زبان طبیعی کمک می‌کنه.", "یادگیری عمیق از شبکه‌های عصبی استفاده می‌کنه.",
            "الان دارم فارسی یاد می‌گیرم.", "می‌تونی کمکم کنی؟",
            "روباه قهوه‌ای سریع می‌پره.", "مهندسی نرم‌افزار به خلاقیت نیاز داره.",
            "امنیت سایبری خیلی مهمه.", "محاسبات ابری همه جا هست.",
            "اینترنت مردم رو به هم وصل می‌کنه.", "تکنولوژی به سرعت پیشرفت می‌کنه.",
            "موسیقی منو خوشحال می‌کنه.", "دوست دارم کتاب بخونم.",
            "قهوه نوشیدنی مورد علاقه منه.", "ورزش برای سلامتی خوبه.",
            "سفر کردن ذهن رو گسترش می‌ده.", "تحصیلات درها رو باز می‌کنه.",
            "خورشید از شرق طلوع می‌کنه.", "آب برای زندگی ضروریه.",
            "پرنده‌ها تو آسمون پرواز می‌کنن.", "گل‌ها در بهار شکوفه می‌دن.",
            "دیروز رفتم خرید.", "داره کتاب می‌خونه.",
            "فردا می‌رسن.", "تو یه شرکت کار می‌کنه.",
            "بچه‌ها تو پارک بازی می‌کنن.", "دانشجوها تو کتابخونه درس می‌خونن.",
            "ماشین خیلی سریعه.", "این خونه قدیمیه.",
            "باتری گوشیم کمه.", "فیلم خوب بود.",
            "باید خرید کنم.", "بریم رستوران.",
            "قطار ظهر حرکت می‌کنه.", "می‌تونی آروم حرف بزنی؟",
            "یکم چای می‌خوام.", "الان داره بارون میاد.",
            "غذا خوشمزه است.", "بعد از کار خسته‌ام.",
            "باید زود بریم.",
        ],
    }

    data = {'text': [], 'language': []}
    for lang, texts in dataset.items():
        data['text'].extend(texts)
        data['language'].extend([lang] * len(texts))

    df = pd.DataFrame(data)

    print(f"✓ Dataset created successfully")
    print(f"  Total samples: {len(df)}")
    print(f"  Languages: {df['language'].nunique()}")

    return df


def evaluate(classifier, test_df):
    """
    Evaluate classifier on a test set and return detailed metrics.

    Args:
        classifier: Trained classifier instance
        test_df (DataFrame): Test DataFrame with 'text' and 'language' columns

    Returns:
        dict: overall_accuracy, per_language_metrics, predictions, true_labels,
              confusion_matrix
    """
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    predictions = []
    true_labels = []

    for _, row in test_df.iterrows():
        pred = classifier.predict(row['text'])
        predictions.append(pred)
        true_labels.append(row['language'])

    correct = sum(p == t for p, t in zip(predictions, true_labels))
    overall_accuracy = correct / len(test_df)

    per_lang_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for pred, true in zip(predictions, true_labels):
        per_lang_metrics[true]['total'] += 1
        if pred == true:
            per_lang_metrics[true]['correct'] += 1

    for lang in per_lang_metrics:
        total = per_lang_metrics[lang]['total']
        corr = per_lang_metrics[lang]['correct']
        per_lang_metrics[lang]['accuracy'] = corr / total if total > 0 else 0

    # Build confusion matrix
    languages = sorted(set(true_labels))
    confusion = {t: defaultdict(int) for t in languages}
    for pred, true in zip(predictions, true_labels):
        confusion[true][pred] += 1

    return {
        'overall_accuracy': overall_accuracy,
        'per_language_metrics': dict(per_lang_metrics),
        'predictions': predictions,
        'true_labels': true_labels,
        'confusion_matrix': confusion,
        'languages': languages,
    }


def print_confusion_matrix(results):
    """
    Print a formatted confusion matrix to stdout.

    Args:
        results (dict): Output of evaluate()
    """
    languages = results['languages']
    confusion = results['confusion_matrix']

    col_w = 10
    header = f"{'':12s}" + "".join(f"{lang:>{col_w}}" for lang in languages)
    print(header)
    print("-" * len(header))

    for true_lang in languages:
        row_str = f"{true_lang:<12s}"
        for pred_lang in languages:
            count = confusion[true_lang].get(pred_lang, 0)
            marker = f"[{count}]" if true_lang == pred_lang else f" {count} "
            row_str += f"{marker:>{col_w}}"
        print(row_str)


def main():
    """Main training function."""

    print("="*60)
    print("LANGUAGE IDENTIFICATION SYSTEM  v1.0.0")
    print("Enhanced Bigram + Trigram Naive Bayes")
    print("="*60)

    Path("models").mkdir(exist_ok=True)

    # 1. Create dataset
    print("\n[STEP 1] Creating dataset...")
    df = create_complete_dataset()

    print(f"\n📊 Dataset Summary:")
    for lang, count in df['language'].value_counts().items():
        print(f"  • {lang}: {count} samples")

    # 2. Split data (80/20 stratified)
    print("\n[STEP 2] Splitting data (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df,
        train_size=0.8,
        stratify=df['language'],
        random_state=42
    )
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # 3. Train enhanced classifier
    print("\n[STEP 3] Training enhanced classifier (bigrams + trigrams)...")
    classifier = EnhancedNaiveBayesLanguageIdentifier()
    classifier.train(train_df)

    # 4. Save model
    model_path = "models/language_model.pkl"
    print(f"\n[STEP 4] Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print("  ✓ Model saved successfully")

    # 5. Evaluate
    print("\n[STEP 5] Evaluating...")
    results = evaluate(classifier, test_df)

    print(f"\n{'='*60}")
    print("📈 RESULTS")
    print("="*60)
    print(f"\n🎯 Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"\n📊 Per-Language Accuracy:")
    print("-"*60)

    for lang in sorted(results['per_language_metrics'].keys()):
        metrics = results['per_language_metrics'][lang]
        acc = metrics['accuracy']
        emoji = "🟢" if acc >= 0.90 else ("🟡" if acc >= 0.75 else "🔴")
        print(f"  {emoji} {lang:12s}: {acc:5.1%}  ({metrics['correct']}/{metrics['total']})")

    # 6. Confusion matrix
    print(f"\n{'='*60}")
    print("🔢 CONFUSION MATRIX  (rows = true, cols = predicted)")
    print("="*60 + "\n")
    print_confusion_matrix(results)

    # 7. Edge case / short-text tests
    print(f"\n{'='*60}")
    print("🧪 TESTING SHORT PHRASES")
    print("="*60 + "\n")

    edge_cases = [
        ("spanish", "Hola"),
        ("spanish", "Buenos días"),
        ("french",  "Bonjour"),
        ("english", "Hello"),
        ("german",  "Guten Tag"),
        ("italian", "Ciao"),
        ("persian", "سلام"),
    ]

    for true_lang, text in edge_cases:
        pred_lang, probs = classifier.predict_with_confidence(text)
        low_conf = probs.pop('low_confidence', False)
        is_correct = pred_lang == true_lang
        emoji = "✅" if is_correct else "❌"
        warning = " ⚠ low-confidence" if low_conf else ""
        print(f"{emoji} '{text}' → {pred_lang} ({probs[pred_lang]:.0%})"
              f" [expected: {true_lang}]{warning}")

    print(f"\n{'='*60}")
    print("✨ TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Model saved to: models/language_model.pkl")
    print("\n🚀 Next steps:")
    print("  python 2_test_model.py")
    print("  python 3_interactive_test.py")


if __name__ == "__main__":
    main()
