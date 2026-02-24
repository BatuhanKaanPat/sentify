import os
import sys
import pickle
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.preprocessing import load_data, preprocess_text, download_nltk_data, simple_tokenizer

def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a single model.
    Returns a dict of metrics and the fitted pipeline.
    """
    print(f"Training {name}...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess_text,
            tokenizer=simple_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),   # unigrams + bigrams ("I like", "not good")
            max_features=15000,
        )),
        ('clf', model)
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - start_time
    
    # Handle potentially string labels
    unique_labels = y_test.unique()
    pos_label = 'positive'
    if 'positive' not in unique_labels:
        if '1' in unique_labels: pos_label = '1'
        elif 1 in unique_labels: pos_label = 1
        elif len(unique_labels) == 2: pos_label = unique_labels[1]

    # Convert to strings for consistent metric calculation if needed
    if isinstance(y_pred[0], str) and not isinstance(y_test.iloc[0], str):
        y_test_converted = y_test.astype(str)
    elif not isinstance(y_pred[0], str) and isinstance(y_test.iloc[0], str):
        y_test_converted = y_test.astype(int)
    else:
        y_test_converted = y_test

    acc = accuracy_score(y_test_converted, y_pred)
    prec = precision_score(y_test_converted, y_pred, pos_label=pos_label, average='binary')
    rec = recall_score(y_test_converted, y_pred, pos_label=pos_label, average='binary')
    f1 = f1_score(y_test_converted, y_pred, pos_label=pos_label, average='binary')
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Inference Time (s)': inference_time
    }, pipeline

def benchmark_models():
    """
    Train and benchmark multiple models.
    Saves results and best model.
    """
    print("Starting benchmarking pipeline...")
    download_nltk_data()
    
    # 1. Load Data
    df = load_data()
    X = df['review']
    y = df['sentiment']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save test data for evaluation
    test_df = pd.DataFrame({'review': X_test, 'sentiment': y_test})
    os.makedirs('data', exist_ok=True)
    test_df.to_csv('data/test.csv', index=False)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear SVM': LinearSVC(random_state=42, dual=False), # dual=False preferred when n_samples > n_features
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Multinomial NB': MultinomialNB()
    }
    
    results = []
    best_f1 = -1
    best_pipeline = None
    best_model_name = ""
    
    for name, model in models.items():
        metrics, pipeline = train_and_evaluate(model, name, X_train, y_train, X_test, y_test)
        results.append(metrics)
        
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_pipeline = pipeline
            best_model_name = name
            
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs('models', exist_ok=True)
    results_df.to_csv('models/model_comparison.csv', index=False)
    print("Results saved to models/model_comparison.csv")
    
    # Save best model
    if best_pipeline:
        print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
        model_path = 'models/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_pipeline, f)
        print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    benchmark_models()
