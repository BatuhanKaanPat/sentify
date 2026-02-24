import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.getcwd())

# preprocess_text is needed for unpickling if it's used in pipeline?
# Yes, because TfidfVectorizer(preprocessor=preprocess_text) refers to the function.
from src.preprocessing import preprocess_text

def evaluate_model():
    """
    Evaluate the trained model.
    1. Load Model
    2. Load Test Data
    3. Predict
    4. Compute Metrics
    5. Plot Confusion Matrix
    """
    model_path = 'models/best_model.pkl'
    test_data_path = 'data/test.csv'
    
    if not os.path.exists(model_path):
        print("Model not found. Please run src/train.py first.")
        return
        
    if not os.path.exists(test_data_path):
        print("Test data not found. Please run src/train.py first.")
        return
        
    print("Loading model and test data...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    df_test = pd.read_csv(test_data_path)
    X_test = df_test['review'] # Raw text now
    y_test = df_test['sentiment']
    
    print("Predicting...")
    # Model pipeline includes preprocessing now
    y_pred = model.predict(X_test)
    
    # Ensure types match
    if isinstance(y_pred[0], str) and not isinstance(y_test.iloc[0], str):
        print("Converting y_test to string to match predictions.")
        y_test = y_test.astype(str)
    elif not isinstance(y_pred[0], str) and isinstance(y_test.iloc[0], str):
         print("Converting y_test to int to match predictions.")
         y_test = y_test.astype(int)
    
    # 4. Compute Metrics
    print("\nEvaluation Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    
    unique_labels = y_test.unique()
    # Handle potentially string labels from OpenML ('positive', 'negative' or '1', '0')
    pos_label = 'positive'
    if 'positive' not in unique_labels:
        # try to guess
        if '1' in unique_labels: pos_label = '1'
        elif 1 in unique_labels: pos_label = 1
        elif len(unique_labels) == 2: pos_label = unique_labels[1] # heuristic
    
    print(f"Using pos_label: {pos_label}")
    
    try:
        precision = precision_score(y_test, y_pred, pos_label=pos_label, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=pos_label, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, average='binary')
    except ValueError:
        # Fallback for weird labels
        print("Warning: Binary metrics failed, switching to weighted average.")
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
