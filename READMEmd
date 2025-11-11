# ML-Based ECG Signal Classification for Enhanced Cardiac Diagnosis

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Implemented-green?logo=xgboost)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📋 Project Overview
This project builds a machine-learning pipeline that classifies ECG (electrocardiogram) records into diagnostic categories such as **Normal**, **Mild Abnormalities**, and **Severe Abnormalities**.  
The dataset contains *pre-extracted ECG features* (intervals, amplitudes, signal statistics).  
Models are trained to detect cardiac irregularities automatically—supporting early diagnosis.

---

## 🧠 Objectives
1. Clean and balance an imbalanced ECG dataset.  
2. Train and compare multiple ML algorithms (Decision Tree, Random Forest, XGBoost).  
3. Evaluate models using accuracy, precision, recall, and F1-score.  
4. Visualize model performance through confusion matrix, ROC curves, and feature importance.  
5. Generate interpretable insights for medical decision-support systems.

---

## ⚙️ Workflow
1. **Data Understanding** → Load and explore ECG feature dataset.  
2. **Preprocessing** → Handle missing values, encode categorical labels, resample classes.  
3. **Scaling** → Standardize numeric features using `StandardScaler`.  
4. **Modeling** → Train Decision Tree, Random Forest, and XGBoost classifiers.  
5. **Evaluation** → Compare metrics and visualize results.  
6. **Reporting** → Export plots and CSV summaries for documentation.

---

## 🧩 Project Structure

ECG_Minor/
│
├─ data/
│ └─ ecg_data.csv
├─ notebooks/
│ ├─ 01_data_preprocessing.ipynb
│ └─ 02_model_training.ipynb
├─ reports/
│ ├─ figures/
│ │ ├─ confusion_matrix.png
│ │ ├─ feature_importance.png
│ │ └─ roc_multiclass.png
│ └─ model_comparison.csv
├─ src/
│ ├─ preprocessing.py
│ ├─ model_training.py
│ └─ evaluation.py
└─ README.md


---

## 🧮 Model Results
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|----------|
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 0.983 | 0.983 | 0.983 | 0.983 |

> **Observation:** Perfect results for Decision Tree and XGBoost indicate near-separable features, typical for feature-engineered ECG datasets.  
> Random Forest is retained as the final model for better generalization and interpretability.

---

## 📊 Visual Outputs
*(Add screenshots or links to your plots)*
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`
- `reports/figures/roc_multiclass.png`

---

## 💡 Insights & Discussion
- The dataset’s ECG features provide strong separability across cardiac classes.  
- Balancing classes improved model fairness and recall.  
- Random Forest explained key signal attributes contributing to cardiac-state prediction.  
- Future enhancement: include raw ECG waveform processing for signal-to-feature extraction.

---

## 🧰 Tech Stack
**Languages:** Python  
**Libraries:** pandas, NumPy, scikit-learn, XGBoost, seaborn, matplotlib  
**Tools:** Jupyter Notebook, Git, PowerShell

---

## 🚀 How to Run
          ```bash
          git clone https://github.com/<your-username>/ECG_Minor.git
          cd ECG_Minor
          pip install -r requirements.txt
          # open notebooks/02_model_training.ipynb and run all cells

📈 Future Work
1. Integrate CNN/LSTM for raw ECG waveform analysis.
2. Build a Streamlit dashboard for real-time predictions.
3. Deploy trained model as a REST API.


👤 Author

Jatin Purushwani

📧 jatin.upskill.84588@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/jatin-purushwani-875432299/  
    GitHub: https://github.com/JatinPurushwani/Spam-Detection          


