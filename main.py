import csv
import nltk
import string
import numpy as np
import pandas as pd
import regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#We need to know performance of our models
from sklearn.metrics import classification_report, accuracy_score
# Download once if needed
nltk.download('all')


stop_words = set(stopwords.words('english'))
#add here words that will be ignored 
#important_words = {'yourself', 'myself', 'himself', 'herself', 'ourselves', 'themselves'}
#stop_words -= important_words
punctuations = set(string.punctuation)

# Remove everything except letters and spaces
def remove_unwanted_characters(text):
    return ''.join([ch for ch in text if ch.isalpha() or ch.isspace()])

# Strip out URLs
def remove_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    return regex.sub(url_pattern, '', text)

# Core text cleaning
def preprocess_text(text):
    text = str(text).lower()
    text = remove_urls(text)
    text = regex.sub(r'\d+', '', text)  # remove digits
    text = remove_unwanted_characters(text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words and word not in punctuations]
    return ' '.join(filtered)

# Apply cleaning only to 'text' column
def pre_processing(df):
    df = df.copy()
    if 'text' in df.columns:
        df['text'] = df['text'].apply(preprocess_text)
    return df

# Save predictions to file
def save_predictions_to_txt(predictions, filename):
    np.savetxt(filename, predictions, fmt='%s')

# Export cleaned data
def save_preprocessed_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Main training and prediction pipeline
def train_test(train_path, test_path):
    processed_file = "preprocessed_data.csv"

    # Load train CSV
    df = pd.read_csv(train_path, sep=",", quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip', engine='python')
    df.columns = [col.strip().lower().replace(';;', '') for col in df.columns]


    # Preprocess text
    df = pre_processing(df)
    save_preprocessed_to_csv(df, processed_file)

    X = df['text']
    y = df['emotion']

    ###Split data into train and test sets, 30% for testing 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

    #VECTORIZE DATA BATA FOR BETTER PERFORMANCE
    tfidf = TfidfVectorizer()
    tfidf_bi = TfidfVectorizer(ngram_range=(1, 2))

    # Define models to compare
    models = [
        ("MNB", Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])),
        ("CNB", Pipeline([('tfidf', tfidf), ('clf', ComplementNB())])),
        ("SVC", Pipeline([('tfidf', tfidf_bi), ('clf', LinearSVC(dual="auto"))])),
        ("DT", Pipeline([('tfidf', tfidf), ('clf', DecisionTreeClassifier(random_state=42))])),
        ("KNN", Pipeline([('tfidf', tfidf), ('clf', KNeighborsClassifier(n_neighbors=5))]))
    ]

    print("Evaluating models...\n")
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))
        save_predictions_to_txt(preds, f"predictions_{name}.txt")

    # Final model trained on all data, SVM
    print("\n Training final model on full dataset...")
    final_model = Pipeline([
        ('tfidf', tfidf_bi),
        ('clf', LinearSVC(dual="auto"))
    ])
    final_model.fit(X, y)

    # Predict on the new testing data
    predict_on_new_text(final_model, test_path)

#Run prediction on a new dataset 
def predict_on_new_text(model, test_path, output_file="predicted_emotions.txt"):
    df_test = pd.read_csv(test_path, header=None, names=['text'], sep=",", engine="python", on_bad_lines="skip")

    # Preprocess text
    df_test['text'] = df_test['text'].apply(preprocess_text)

    predictions = model.predict(df_test['text'])
    df_test['predicted_emotion'] = predictions

    df_test.to_csv(output_file, index=False)
    print(f"\n Predictions saved to {output_file}")
    print(df_test[['text', 'predicted_emotion']].head())

#start
train_test("train_emotion.csv", "Test.csv")
