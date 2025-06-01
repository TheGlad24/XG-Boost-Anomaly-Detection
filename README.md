# XGBoost Anomaly Detection on UNSW-NB15 Dataset

This project applies the XGBoost algorithm for anomaly detection using the **UNSW-NB15 dataset**, a modern labeled dataset built for network intrusion detection research. It demonstrates how a machine learning model can effectively detect attacks in **highly imbalanced data**, where traditional accuracy can be misleading.

---

## 📁 Datasets Used

The dataset consists of **four raw CSV files**:

- `UNSW-NB15_1.csv`
- `UNSW-NB15_2.csv`
- `UNSW-NB15_3.csv`
- `UNSW-NB15_4.csv`

Each file contains labeled network flow records (totaling over 2 million), including:

- IP addresses, ports, byte counts
- Duration, protocol types, service flags
- Labels: "Normal" vs specific attack types (e.g., DoS, Exploits, Worms)

These files are split for collection purposes but are structurally identical.

---

## 🔄 Data Preprocessing

The script `src/preprocess.py` performs the following pipeline:

1. **Combines** all four CSV files into a single dataset.
2. **Cleans** rows with missing or corrupted values.
3. **Encodes** categorical features using label encoding.
4. **Normalizes** continuous numeric features.
5. Saves the result as:  
   ➤ `data/processed/combined_cleaned.csv`

This final dataset is used for training and evaluation.

---

## ⚙️ Model: XGBoost

We use **XGBoost**, a gradient-boosted decision tree algorithm known for its high performance on structured/tabular data.

---

## 📊 XGBoost Results

| Metric        | Value     |
|---------------|-----------|
| **Accuracy**  | 41.27%    |
| **Precision** | 79.54%    |
| **Recall**    | 83.69%    |
| **F1 Score**  | 81.56%    |
| **AUC-ROC**   | 90.12%    |

---

## 🤔 Why F1 Score Matters More Than Accuracy

This project deals with a **class-imbalanced problem** — the dataset contains **far more normal network traffic** than malicious activity. In such cases:

- **Accuracy is misleading**:
  > A model can be 95% "accurate" by always predicting **"normal"**, yet it may completely miss actual attacks.

- **Precision** tells us: *Of all predicted attacks, how many were real?*
- **Recall** tells us: *Of all actual attacks, how many did we catch?*
- **F1 Score** balances these — it’s the harmonic mean of precision and recall.

In **intrusion detection**, this matters greatly:
- Missing an attack (false negative) can be **catastrophic**
- Flagging some false positives is acceptable if real threats are caught

---

## 🧠 What Our Model Achieved

Despite **low accuracy** (41.27%) — expected due to imbalance — our model:

- Correctly caught **most real attacks** (recall = 83.69%)
- Made **very few false attack predictions** (precision = 79.54%)
- Balanced performance with an **F1-score of 81.56%**
- Scored **AUC-ROC of 90.12%**, meaning good class separation

➡️ This makes it **practically useful** for real-world anomaly detection.

---

## 📂 Project Structure

📁 src/
│ ├── preprocess.py # Dataset loading and cleaning
│ ├── xgboost_model.py # XGBoost training and evaluation

📁 data/ # Large CSVs ignored in GitHub
│ └── raw/ # UNSW-NB15_*.csv files
│ └── processed/ # combined_cleaned.csv

📁 models/ # Trained XGBoost model
📁 outputs/ # Plots, confusion matrix, metrics
├── requirements.txt # Install dependencies
└── README.md # You're reading it

yaml
Copy
Edit

---

## 🚫 GitHub File Limit Note

Due to GitHub's 100MB file limit, large CSV files (`data/raw/`, `data/processed/`) are excluded from this repository.

To reproduce the project:
1. Download the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. Place the 4 `.csv` files in `data/raw/`
3. Run:
   ```bash
   python src/preprocess.py
   python src/xgboost_model.py
✅ Summary
This project proves that:

Accuracy is not the right metric for imbalanced anomaly detection

F1 Score and AUC-ROC are better reflections of model quality

XGBoost can detect intrusions with high precision and recall even in real-world, noisy, imbalanced network data

