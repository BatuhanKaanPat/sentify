import os
import sys
import time
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure src can be imported
sys.path.append(os.getcwd())

def load_model():
    """
    Load the pre-trained DistilBERT model for sentiment analysis.
    Uses CPU (device=-1).
    """
    print("Loading DistilBERT model...")
    # device=-1 for CPU
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1 
    )
    return sentiment_pipeline

def run_inference(model, data):
    """
    Run batch inference on the provided data.
    """
    print(f"Running inference on {len(data)} samples...")
    start_time = time.time()
    
    # Truncate text to avoiding token length issues (DistilBERT max is 512 tokens)
    # Simple truncation by character length as a heuristic or let pipeline handle it with truncation=True
    # Pipeline handles truncation if specified.
    
    # Ensure list of strings
    input_data = [str(x) for x in data.tolist()]
    
    results = model(input_data, truncation=True, max_length=512, batch_size=16)
    
    inference_time = time.time() - start_time
    
    # Extract labels
    # Model returns [{'label': 'POSITIVE', 'score': ...}, ...]
    predictions = [res['label'] for res in results]
    
    # Map to 'positive'/'negative' to match our dataset
    # DistilBERT SST-2 uses 'POSITIVE'/'NEGATIVE'
    predictions_mapped = [p.lower() for p in predictions]
    
    return predictions_mapped, inference_time

def evaluate_transformer():
    """
    Load test data, run inference, compute metrics, and update comparison CSV.
    """
    test_data_path = 'data/test.csv'
    comparison_path = 'models/model_comparison.csv'
    
    if not os.path.exists(test_data_path):
        print("Test data not found. Please run src/train.py first.")
        return
        
    print("Loading test data...")
    df_test = pd.read_csv(test_data_path)
    X_test = df_test['review']
    y_test = df_test['sentiment']
    
    # Load model
    model = load_model()
    
    # Run inference
    y_pred, inference_time = run_inference(model, X_test)
    
    # Compute metrics
    unique_labels = y_test.unique()
    # Map model output to match ground truth
    # DistilBERT pipeline returns 'POSITIVE'/'NEGATIVE'
    # Project dataset might be 'positive'/'negative' or '1'/'0'.
    # We normalized to 'positive'/'negative' in run_inference.
    
    # Ensure y_test is consistent
    y_test = y_test.astype(str).str.lower().str.strip()
    y_pred = [str(p).lower().strip() for p in y_pred]
    
    print(f"Unique y_test labels: {set(y_test)}")
    print(f"Unique y_pred labels: {set(y_pred)}")
    
    # Check for unexpected labels
    valid_labels = {'positive', 'negative'}
    if not set(y_test).issubset(valid_labels) or not set(y_test).issubset(valid_labels):
        print("Warning: Found unexpected labels. Mapping common variations.")
        label_map = {'1': 'positive', '0': 'negative', 'pos': 'positive', 'neg': 'negative'}
        y_test = y_test.replace(label_map)
        y_pred = [label_map.get(p, p) for p in y_pred]
        print(f"Mapped y_test unique: {set(y_test)}")
        print(f"Mapped y_pred unique: {set(y_pred)}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='positive', average='binary')
    recall = recall_score(y_test, y_pred, pos_label='positive', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='positive', average='binary')
    
    metrics = {
        'Model': 'DistilBERT',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time (s)': 0.0, # Pre-trained
        'Inference Time (s)': inference_time
    }
    
    print("\nDistilBERT Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # Append to comparison CSV
    if os.path.exists(comparison_path):
        df_comparison = pd.read_csv(comparison_path)
        # Remove existing if any
        if 'DistilBERT' in df_comparison['Model'].values:
            df_comparison = df_comparison[df_comparison['Model'] != 'DistilBERT']
        
        # Concat new row
        new_row = pd.DataFrame([metrics])
        df_comparison = pd.concat([df_comparison, new_row], ignore_index=True)
    else:
        df_comparison = pd.DataFrame([metrics])
        
    # Save updated CSV
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\nUpdated model comparison:\n{df_comparison.to_string(index=False)}")
    print(f"Results saved to {comparison_path}")

if __name__ == "__main__":
    evaluate_transformer()
