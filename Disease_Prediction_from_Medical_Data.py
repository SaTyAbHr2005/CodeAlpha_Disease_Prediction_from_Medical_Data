import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

def create_breast_cancer_csv():
    """Create breast_cancer.csv from sklearn if it doesn't exist or has wrong format"""
    import os
    try:
        # Check if file exists and has correct format
        if os.path.exists('breast_cancer.csv'):
            test_df = pd.read_csv('breast_cancer.csv')
            if 'target' in test_df.columns and 'diagnosis' not in test_df.columns:
                return  # File is already correct
        
        print("Creating correct breast_cancer.csv from sklearn dataset...")
        breast_cancer = load_breast_cancer(as_frame=True)
        breast_df = breast_cancer.frame
        breast_df.to_csv('breast_cancer.csv', index=False)
        print("breast_cancer.csv created successfully!")
    except Exception as e:
        print(f"Error creating breast_cancer.csv: {e}")

class DiseasePredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        
    def load_csv_data(self, filename):
        """Load data from CSV file with proper preprocessing"""
        try:
            df = pd.read_csv(filename)
            print(f"Loading {filename} - Original shape: {df.shape}")
            
            # Dataset-specific preprocessing
            if filename == 'heart_disease.csv':
                # Drop unnecessary columns
                columns_to_drop = []
                if 'id' in df.columns:
                    columns_to_drop.append('id')
                if 'dataset' in df.columns:
                    columns_to_drop.append('dataset')
                if columns_to_drop:
                    df = df.drop(columns_to_drop, axis=1)
                
                # Rename 'num' to 'target' and convert to binary
                if 'num' in df.columns:
                    df = df.rename(columns={'num': 'target'})
                    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
                    print("Converted 'num' to binary 'target'")
                
                # Convert categorical variables to numeric
                if 'sex' in df.columns and df['sex'].dtype == 'object':
                    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})
                
                # Handle other categorical columns
                categorical_columns = df.select_dtypes(include=['object']).columns
                categorical_columns = [col for col in categorical_columns if col != 'target']
                for col in categorical_columns:
                    df[col] = pd.Categorical(df[col]).codes
                    
            elif filename == 'diabetes.csv':
                # Rename 'Outcome' to 'target' if exists
                if 'Outcome' in df.columns:
                    df = df.rename(columns={'Outcome': 'target'})
                    print("Renamed 'Outcome' to 'target'")
                    
            elif filename == 'breast_cancer.csv':
                # Handle both formats: sklearn format and original CSV format
                print(f"Original columns: {list(df.columns)}")
                
                # Drop unnecessary columns
                columns_to_drop = []
                if 'id' in df.columns:
                    columns_to_drop.append('id')
                if 'Unnamed: 32' in df.columns:
                    columns_to_drop.append('Unnamed: 32')
                
                # Handle empty columns
                empty_cols = [col for col in df.columns if df[col].isna().all()]
                columns_to_drop.extend(empty_cols)
                
                if columns_to_drop:
                    df = df.drop(columns_to_drop, axis=1)
                    print(f"Dropped columns: {columns_to_drop}")
                
                # Convert diagnosis to target if present
                if 'diagnosis' in df.columns:
                    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
                    df = df.drop('diagnosis', axis=1)
                    print("Converted 'diagnosis' (M/B) to 'target' (1/0)")
                
                # Ensure all columns are numeric except target
                for col in df.columns:
                    if col != 'target' and df[col].dtype == 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any remaining missing values
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)
            
            if initial_rows != final_rows:
                print(f"Removed {initial_rows - final_rows} rows with missing values")
            
            # Final validation
            if 'target' not in df.columns:
                print(f"Error: 'target' column not found in {filename}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            if df.shape[0] == 0:
                print(f"Error: No samples remaining after preprocessing {filename}")
                return None
            
            # Ensure target is numeric
            df['target'] = pd.to_numeric(df['target'], errors='coerce')
            df = df.dropna()  # Remove any rows where target conversion failed
            
            print(f"Final shape: {df.shape}")
            print(f"Target distribution: {dict(df['target'].value_counts())}")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            if filename == 'breast_cancer.csv':
                print("Attempting to create breast_cancer.csv...")
                create_breast_cancer_csv()
                try:
                    return self.load_csv_data(filename)  # Recursive call
                except Exception as e:
                    print(f"Failed to create/load {filename}: {e}")
                    return None
            return None
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def prepare_data(self, df, dataset_name):
        """Prepare data for training"""
        df = shuffle(df, random_state=42)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Check if we have enough samples
        if len(X) < 10:
            print(f"Error: Not enough samples ({len(X)}) for training")
            return None, None, None, None
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        # Ensure we have both classes
        if len(class_counts) < 2:
            print(f"Error: Only one class present in target variable")
            return None, None, None, None
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"Error in train_test_split: {e}")
            # Try without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scalers[dataset_name] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_classifiers(self):
        """Define classifiers to use"""
        classifiers = {
            'SVM': SVC(kernel='linear', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        return classifiers
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, dataset_name):
        """Train multiple classifiers and evaluate performance"""
        results = {}
        classifiers = self.get_classifiers()
        
        print(f"\n{'='*60}")
        print(f"Training models for {dataset_name.upper()} dataset")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"{'='*60}")
        
        for name, clf in classifiers.items():
            print(f"Training {name}...")
            
            try:
                # Train the model
                clf.fit(X_train, y_train)
                
                # Make predictions
                y_pred = clf.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': clf,
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred),
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"{name} completed - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Store models for this dataset
        self.models[dataset_name] = results
        return results
    
    def display_results(self, results, dataset_name):
        """Display detailed results"""
        if not results:
            print(f"No results to display for {dataset_name}")
            return
            
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"\n#{i} {name}")
            print(f"{'-' * 40}")
            print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print("Classification Report:")
            print(result['classification_report'])
    
    def predict_new_patient(self, dataset_name, model_name, patient_data):
        """Predict disease for new patient data"""
        if dataset_name not in self.models:
            return f"No trained models found for {dataset_name}"
            
        if model_name not in self.models[dataset_name]:
            return f"Model {model_name} not found for {dataset_name}"
        
        try:
            # Scale the input data
            scaler = self.scalers[dataset_name]
            patient_data_scaled = scaler.transform([patient_data])
            
            # Make prediction
            model = self.models[dataset_name][model_name]['model']
            prediction = model.predict(patient_data_scaled)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(patient_data_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = None
            
            return {
                'prediction': prediction,
                'risk_level': 'HIGH RISK' if prediction == 1 else 'LOW RISK',
                'confidence': confidence,
                'model_used': model_name
            }
        except Exception as e:
            return f"Error making prediction: {e}"

def main():
    """Main function to run the disease prediction system"""
    
    print("DISEASE PREDICTION SYSTEM STARTING...")
    print("=" * 80)
    
    # Ensure breast cancer CSV exists and is in correct format
    create_breast_cancer_csv()
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Dataset configurations
    datasets = {
        'heart_disease': {
            'file': 'heart_disease.csv',
            'name': 'Heart Disease',
            'description': 'Predict heart disease based on clinical features'
        },
        'diabetes': {
            'file': 'diabetes.csv', 
            'name': 'Diabetes',
            'description': 'Predict diabetes based on medical measurements'
        },
        'breast_cancer': {
            'file': 'breast_cancer.csv',
            'name': 'Breast Cancer',
            'description': 'Predict breast cancer malignancy'
        }
    }
    
    all_results = {}
    
    # Process each dataset
    for dataset_key, dataset_info in datasets.items():
        print(f"\nProcessing {dataset_info['name']}...")
        print(f"{dataset_info['description']}")
        
        # Load data
        df = predictor.load_csv_data(dataset_info['file'])
        if df is None:
            print(f"Skipping {dataset_info['name']} - could not load data")
            continue
            
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
        
        # Prepare and train
        data_split = predictor.prepare_data(df, dataset_key)
        if any(x is None for x in data_split):
            print(f"Skipping {dataset_info['name']} - data preparation failed")
            continue
            
        X_train, X_test, y_train, y_test = data_split
        results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test, dataset_key)
        all_results[dataset_key] = results
        
        # Display results
        predictor.display_results(results, dataset_info['name'])
    
    # Overall summary
    print(f"\nFINAL SUMMARY")
    print("=" * 80)
    
    for dataset_key, results in all_results.items():
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            dataset_name = datasets[dataset_key]['name']
            print(f"{dataset_name:15} | Best: {best_model[0]:18} | Accuracy: {best_model[1]['accuracy']:.4f}")
    
    return predictor

def example_predictions(predictor):
    """Show example predictions"""
    print(f"\nEXAMPLE PREDICTIONS")
    print("=" * 80)
    
    examples = {
        'heart_disease': {
            'data': [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6],
            'description': '63-year-old male with chest pain'
        },
        'diabetes': {
            'data': [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            'description': '50-year-old with 6 pregnancies, high glucose'
        },
        'breast_cancer': {
            'data': [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
            'description': 'Breast tissue sample with concerning features'
        }
    }
    
    for dataset_name, example in examples.items():
        if dataset_name in predictor.models and predictor.models[dataset_name]:
            # Use the best performing model
            best_model_name = max(predictor.models[dataset_name].items(), 
                                key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient(dataset_name, best_model_name, example['data'])
            
            if isinstance(result, dict):
                print(f"\n{dataset_name.replace('_', ' ').title()} Prediction:")
                print(f"   Patient: {example['description']}")
                print(f"   Result: {result['risk_level']}")
                print(f"   Model: {result['model_used']}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']:.2%}")
            else:
                print(f"\n{dataset_name.replace('_', ' ').title()}: {result}")

# Run everything
if __name__ == "__main__":
    # Run main training
    predictor = main()
    
    # Show example predictions
    example_predictions(predictor)
    
    print(f"\nDISEASE PREDICTION SYSTEM COMPLETE!")
    print("=" * 80)
    print("All models trained and ready for predictions")
    print("\nUSAGE EXAMPLE:")
    print("result = predictor.predict_new_patient('heart_disease', 'SVM', [63,1,1,145,233,1,2,150,0,2.3,3,0,6])")
    print("print(result['risk_level'], result['confidence'])")
