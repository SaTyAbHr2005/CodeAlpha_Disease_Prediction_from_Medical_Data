# 🩺 Disease Prediction System

This project uses machine learning to predict the presence of different diseases from medical records.  
It focuses on three major conditions:  
- ❤️ Heart Disease  
- 💉 Diabetes  
- 🎗️ Breast Cancer  

## 📊 Datasets
- **Heart Disease**: Patient data including age, sex, chest pain type, blood pressure, cholesterol, etc.  
- **Diabetes**: Pima Indians dataset with features like glucose level, BMI, pregnancies, etc.  
- **Breast Cancer**: Wisconsin dataset containing 30 numerical tumor-related features.  

- Target variable: `target` (1 = disease, 0 = no disease)  
- Features include:  
  - Demographics (age, sex)  
  - Clinical measures (BP, glucose, cholesterol, heart rate)  
  - Diagnostic tests (tumor radius, texture, smoothness, etc.)  

## ⚙️ Data Preprocessing
Steps applied to all datasets:  
- Removal of irrelevant or missing-value columns  
- Conversion of categorical values to numeric codes  
- Standardization of continuous features  
- Train-test split (80:20) with stratification  
- Auto-creation of `breast_cancer.csv` if not available  

## 🤖 Models Implemented
- Support Vector Machine (SVM)  
- Logistic Regression  
- Random Forest Classifier  

Each dataset is trained on all models, and the best-performing model is recorded.  

## 📈 Evaluation
The following metrics are calculated for each model:  
- Accuracy  
- Precision, Recall, and F1-score  
- Classification report and confusion matrix  

A final summary table shows the **best model per disease** with accuracy scores.  

## 🏆 Sample Results
| Disease        | Best Model          | Accuracy |
|----------------|---------------------|----------|
| Heart Disease  | Random Forest       | 0.89     |
| Diabetes       | SVM                 | 0.81     |
| Breast Cancer  | Logistic Regression | 0.95     |

👉 Each condition has a different best model, showing the importance of dataset-specific tuning.  

## 🚀 Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disease-prediction-system.git
   cd disease-prediction-system

2. Install required libraries:
   ```bash
   pip install -r requirements.txt


3. Place datasets (heart_disease.csv, diabetes.csv) in the project folder.
   (The script will auto-generate breast_cancer.csv if missing.)

4. Train and evaluate models:

   ```bash
   python Disease_Prediction_from_Medical_Data.py


5. Try the interactive demo:

    ```bash
    python demo_predictions.py

## 🔮 Future Scope

- Hyperparameter tuning for better performance

- Integration with deep learning models (e.g., ANN)

- Real-time prediction dashboard (Streamlit / Flask)

- Explainability using SHAP values

- Expansion to other diseases/datasets
