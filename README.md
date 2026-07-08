# TASK 3: EMAIL SPAM CLASSIFIER

## 📋 Project Overview
This repository contains an intermediate-level Machine Learning project developed as part of the **CODTECH IT SOLUTIONS** Virtual Internship. The application utilizes natural language processing (NLP) techniques and a probabilistic classification model to automatically categorize incoming text messages and emails into either **Ham (Safe)** or **Spam (🚨 Alert)**.

---

## 🛠️ Technical Implementation & Architecture
The system pipeline consists of the following core engineering steps:

1. **Data Preprocessing & Encoding**
   - Implements an offline standalone mock dataframe designed to hold representative text formats for predictable vector building without network rate-limit restrictions.
   - Vectorizes target textual responses into categorical binary mappings (`ham: 0`, `spam: 1`).

2. **Feature Extraction (TF-IDF)**
   - Utilizes `TfidfVectorizer` to convert unstructured text strings into numerical matrix arrays.
   - Applies case normalization, lowercase conversions, and filters out common English `stop_words` to improve focus on spam-signal indicators.

3. **Machine Learning Model**
   - Implements a **Multinomial Naive Bayes (`MultinomialNB`)** classifier, which is highly optimal for discrete text classification parameters.
   - Splits data partitions cleanly into stratified Train/Test arrays to validate model metrics objectively.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python installed on your local operating machine along with the required execution frameworks:
```bash
pip install pandas scikit-learn numpy
