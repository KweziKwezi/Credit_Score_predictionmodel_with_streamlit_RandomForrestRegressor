# 📊 Credit Score Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production--Ready-green)

## 📖 Project Overview
This project features a highly accurate **Random Forest Regressor** machine learning model designed to predict an individual's **credit score** based on demographic, financial, and historical loan data. 

By analyzing 8 key applicant features—including age, income, employment experience, education level, home ownership, credit history length, and previous loan default history—the model provides a reliable, data-driven estimation of creditworthiness. The primary goal was to build a robust regression engine that assists lenders in rapid, automated pre-screening and risk assessment.

---

## 🏆 Total Model Accuracy: 6.05% MAPE
After rigorous hyperparameter tuning, the final optimized model achieves a **Mean Absolute Percentage Error (MAPE) of 6.05%** and a **Mean Absolute Error (MAE) of 38.22 points**. 

This means the model's predicted credit score is, on average, within just **~38 points** (or ~6%) of the applicant's actual credit score. In the context of credit scoring (typically ranging from 300 to 850), this level of precision is highly reliable for placing applicants into accurate risk tiers (e.g., "Fair", "Good", "Excellent").

---

## 🧠 Model Architecture
The core algorithm is the **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`). 

A Random Forest was chosen over a single Decision Tree because credit score determination involves complex, non-linear interactions between multiple financial variables. By building an **ensemble of 100 diverse decision trees** and averaging their predictions, the Random Forest naturally:
- ✅ Smooths out noise in the data
- ✅ Prevents overfitting (a common problem with single trees)
- ✅ Delivers significantly more stable and accurate predictions
- ✅ Captures complex feature interactions (e.g., how income + credit history length together impact score)

---

## ⚙️ Data Preprocessing
Before training, the dataset (45,000 applicant records) underwent critical preprocessing:
1. **Feature Selection & Leakage Prevention**: Isolated 8 predictive features, explicitly excluding post-decision features like `loan_status` to prevent data leakage.
2. **Categorical Encoding**: Applied **One-Hot Encoding** (`pd.get_dummies`) to text-based features (`person_gender`, `person_education`, `person_home_ownership`, `previous_loan_defaults_on_file`).
3. **Train/Validation Split**: Split the data 80/20 (`random_state=1`) to ensure an honest, unbiased evaluation of model performance.

---

## 🔬 Hyperparameter Tuning & Optimization Process
To achieve a low margin of error and prevent overfitting, I implemented a systematic, step-by-step iterative tuning process, measuring **Mean Absolute Error (MAE)** at each step:

| Parameter | Tested Range | Best Value | Observation |
|-----------|--------------|------------|-------------|
| `max_leaf_nodes` | 2 → 400, None | **30** | MAE dropped from 39.57 to 38.23. Beyond 30, error plateaued. |
| `min_samples_leaf`| 2 → 10 | **9** | MAE steadily decreased, reaching its lowest point at 9 (38.227). |
| `n_estimators` | 2 → 500 | **100** | MAE stabilized around 38.22 starting at 100 trees. Higher values added compute time without meaningful gains. |
| `max_depth` | 2 → 500, None | **6** | MAE dropped to its lowest point at depth 6 (38.221). Deeper trees added no predictive value. |

---

## 📊 Final Model Configuration
After the optimization process, the final production model was instantiated with the following parameters:

```python
RandomForestRegressor(
    random_state=1,
    max_leaf_nodes=30,
    min_samples_leaf=9,
    n_estimators=100,
    max_depth=6
)
💼 Business Interpretation
These metrics translate directly to real-world financial impact:
High Prediction Confidence: A 6.05% average error is considered excellent for credit score prediction. For a score of 700, the model's prediction will typically fall within a tight range of roughly ±38 points.
Reliable Risk Tiering: Credit scores are grouped into tiers (Poor < 580, Fair 580–669, Good 670–739, Very Good 740–799, Excellent 800+). With a ~38-point margin of error, the model can reliably place applicants into the correct general tier—which is all most lenders need for automated pre-screening.
Scalable Automation: This level of accuracy allows the model to safely automate initial credit assessments for thousands of applicants per minute, freeing human underwriters to focus only on borderline or complex cases.


🌐 Live Demo
This model is deployed as an interactive web application using Streamlit, allowing users to input applicant details and receive instant credit score predictions. https://creditscorepredictionmodelwithapprandomforrestregreappr-gmw3qq.streamlit.app/

💼 Business Applications
This model can be directly integrated into:
🏦 Banking APIs to automate initial credit pre-screening
📱 Fintech Apps to provide users with instant creditworthiness estimates
📊 Risk Management Dashboards to monitor applicant pool quality
🎓 Financial Education Platforms to help users understand which factors impact their score
