import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/combined_cleaned.csv"
MODEL_SAVE_PATH = "models/xgboost_model.json"
METRICS_CSV_PATH = "outputs/xgboost_metrics.csv"
CONFUSION_PLOT_PATH = "outputs/plots/confusion_matrix.png"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(set(y_train)),
        eval_metric='mlogloss',
        tree_method='hist',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=150,
        verbosity=1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    # Classification report
    print("\n[Classification Report]")
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(os.path.dirname(CONFUSION_PLOT_PATH), exist_ok=True)
    plt.savefig(CONFUSION_PLOT_PATH)
    plt.close()

    # Extra summary metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')

    print(f"\n[Overall Metrics]")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1 Score : {f1:.4f}")

    # Save summary metrics to CSV
    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
        'Value': [acc, prec, rec, f1]
    })
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

def main():
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training XGBoost model...")
    model = train_xgboost(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print(f"Saving model to {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
