import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import re

# === 1. Adat előkészítése ===
def extract_email_features(email):
    """Egy e-mail szövegéből jellemzők kinyerése."""
    features = {}

    # Hossza szavakban
    words = email.split()
    features['word_count'] = len(words)

    # Szóismétlés
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0

    # Kulcsszavak aránya
    keywords = {"discount", "offer", "winner", "free", "urgent"}  # Reklám kulcsszavak
    features['keyword_ratio'] = sum(1 for word in words if word.lower() in keywords) / len(words) if words else 0

    # Több címzett
    features['multiple_recipients'] = 1 if "," in email else 0

    # Tárgy hosszának vizsgálata
    features['subject_length'] = len(re.findall(r'Subject: (.+)', email))

    # HTML tartalom
    features['has_html'] = 1 if "<html>" in email.lower() else 0

    # Szófajgyakoriság (egyszerűsített: csak számok és írásjelek)
    features['punctuation_ratio'] = sum(1 for char in email if char in "!?.") / len(email) if len(email) > 0 else 0

    # Linkek száma
    features['link_count'] = len(re.findall(r'http[s]?://', email))

    # Speciális karakterek száma
    features['special_char_count'] = sum(1 for char in email if not char.isalnum() and char not in " ")

    # Formázás gyakorisága
    features['formatting_count'] = len(re.findall(r'<[^>]+>', email))

    # Melléklet
    features['has_attachment'] = 1 if "attachment" in email.lower() else 0

    # Spam szavak aránya
    spam_words = {"lottery", "cash", "prize", "urgent", "win"}
    features['spam_word_ratio'] = sum(1 for word in words if word.lower() in spam_words) / len(words) if words else 0

    # Írásjelek gyakorisága
    features['punctuation_ratio'] = sum(1 for char in email if char in ",;!?@#$%") / len(email) if len(email) > 0 else 0

    # Nagybetűs szavak aránya
    features['uppercase_ratio'] = sum(1 for word in words if word.isupper()) / len(words) if words else 0

    return features

# === 2. Adatbázis létrehozása ===
def create_email_dataset():
    """30 db tanítólevél generálása (manuálisan meghatározott címekkel)."""
    emails = [
        "Subject: Winner! You have won a lottery. Click http://scam.com to claim. <html>",
        "Subject: Important meeting scheduled. Please check attached file.",
        "Subject: Special offer just for you! Discounted prices on electronics.",
        "Urgent: Update your account details at http://bank.com.",
        "Normal business correspondence with attached invoice.",
        "Casual chat with no special content.",
        "Subject: URGENT: Claim your prize now.",
        "<html>Formatted newsletter with multiple links and images.",
        "Attachment includes all details for the project discussed.",
        "Subject: Free vacation offer! Act now.",
        # Add more examples...
    ]
    labels = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 1: spam, 0: not spam

    data = pd.DataFrame([extract_email_features(email) for email in emails])
    data['label'] = labels
    return data

# === 3. Modell tanítása és kiértékelése ===
def train_and_evaluate(data):
    """Naiv Bayes algoritmus alkalmazása az adatokon."""
    X = data.drop(columns=['label'])
    y = data['label']

    # Adatok szétválasztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Skálázás (opcionális)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modell tanítása
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Előrejelzés és értékelés
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model

# === 4. Korreláció számítása ===
def calculate_correlations(data):
    """Korreláció számítása az összes paraméter között."""
    correlation_matrix = data.corr()
    print("Correlation Matrix:\n", correlation_matrix)

# === Főprogram ===
data = create_email_dataset()
model = train_and_evaluate(data)
calculate_correlations(data)
