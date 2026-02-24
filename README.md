# Sentify: Sentiment Analysis SaaS MVP

## Project Description
Sentify is a production-ready Sentiment Analysis SaaS MVP. It classifies text as positive or negative using a Logistic Regression model trained on the IMDb movie reviews dataset. The project emphasizes modular code structure, reproducibility, and ease of deployment.

## Folder Structure
```
sentify/
│
├── data/           # Raw and processed data
├── models/         # Trained models and evaluation plots
├── notebooks/      # Jupyter notebooks for experimentation
├── src/            # Source code
│   ├── preprocessing.py  # Data loading and cleaning
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Model evaluation
│
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1.  **Clone the repository** (if applicable) or navigate to the project root.
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Training

To train the model and save artifacts:

```bash
python src/train.py
```

This will:
1.  Download the IMDb dataset automatically.
2.  Preprocess the text data.
3.  Train a Logistic Regression model using TF-IDF features.
4.  Save the trained model to `models/logistic_sentiment_model.pkl`.
5.  Evaluate the model and save the confusion matrix to `models/confusion_matrix.png`.

## Example Output Metrics

After training, the script will output metrics similar to:
-   Accuracy: ~0.89
-   Precision: ~0.88
-   Recall: ~0.90
-   F1 Score: ~0.89

## Future Improvements

-   **Deep Learning**: Implement BERT or RoBERTa for improved accuracy.
-   **API Deployment**: Wrap the model in a FastAPI or Flask application.
-   **Dockerization**: Containerize the application for easier deployment.
-   **CI/CD**: Add GitHub Actions for automated testing and deployment.
