import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. OFFLINE DATASET CREATION
# ==========================================
def load_data():
    """Provides a local standalone dataset to bypass HTTP 429 errors."""
    print("🔄 Creating offline dataset...")
    
    # Sample balanced collection of typical Ham and Spam messages
    data = {
        'label': [
            'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
            'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'
        ],
        'message': [
            'Hey, are we still meeting up for dinner tonight at 7?',
            'WINNER! You have won a free cash prize of $5000. Call now!',
            'Can you send me the lecture notes from yesterday class?',
            'URGENT: Your bank account has been locked. Click here to verify.',
            'Let me know when you are free for a quick phone call.',
            'FREE ringtones! Text standard rates apply to claim yours today.',
            'Do you want to grab some coffee later this afternoon?',
            'Guaranteed credit card approval! No credit check required.',
            'Hi mom, I will be home a bit late today, do not wait for me.',
            'CONGRATULATIONS! Your mobile number won a luxury cruise holiday!',
            'Just checking in to see how your project assignment is going.',
            'Double your income working from home! Click this link right now.',
            'Are you available to help me move this weekend or busy?',
            'Get cheap deals on luxury watches! 90% off for today only.',
            'Thanks for the birthday wishes, hope you are doing well!',
            'Claim your inheritance money from the royal family lottery.'
        ]
    }
    
    df = pd.DataFrame(data)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# ==========================================
# 2. MODEL TRAINING PIPELINE
# ==========================================
def train_spam_classifier():
    df = load_data()
    print(f"✅ Offline data created successfully. Total rows: {len(df)}")
    print(df['label'].value_counts())
    print("-" * 50)

    X = df['message']
    y = df['label_num']

    # Using smaller test split due to smaller sandbox dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("🔤 Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("🤖 Training the Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("✅ Training complete!")
    print("-" * 50)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, vectorizer

# ==========================================
# 3. LIVE PREDICTION FUNCTION
# ==========================================
def predict_email(text, model, vectorizer):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text)[0]
    
    result = "🚨 SPAM" if prediction == 1 else "🍏 HAM (Safe)"
    confidence = probability[prediction] * 100
    
    print(f"\n📩 Email Text: '{text}'")
    print(f"🔍 Result: {result} ({confidence:.2f}% confidence)")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    trained_model, tfidf_vectorizer = train_spam_classifier()
    
    print("\n--- Testing Custom Examples ---")
    
    sample_1 = "Hey! Are we still meeting for lunch today at 1 PM? Let me know."
    predict_email(sample_1, trained_model, tfidf_vectorizer)
    
    sample_2 = "CONGRATULATIONS! You have won a $1000 Walmart gift card. Click here to claim your cash reward now!!!"
    predict_email(sample_2, trained_model, tfidf_vectorizer)