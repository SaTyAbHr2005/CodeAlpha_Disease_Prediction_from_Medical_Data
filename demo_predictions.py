# demo_predictions.py - Interactive Disease Prediction Demo
from task4 import main
import pandas as pd

def run_demo():
    """Run interactive demo of disease prediction system"""
    
    print("DISEASE PREDICTION SYSTEM DEMO")
    print("=" * 50)
    
    # Train models first
    print("Loading and training models...")
    try:
        predictor = main()
        print("Models trained successfully!")
    except Exception as e:
        print(f"Error training models: {e}")
        return
    
    # Interactive menu loop
    while True:
        print("\nCHOOSE A PREDICTION TYPE:")
        print("1. Heart Disease Prediction")
        print("2. Diabetes Prediction") 
        print("3. Breast Cancer Prediction")
        print("4. Show Input Formats")
        print("5. Quick Examples")
        print("6. Exit")
        print("=" * 40)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            heart_disease_demo(predictor)
        elif choice == "2":
            diabetes_demo(predictor)
        elif choice == "3":
            breast_cancer_demo(predictor)
        elif choice == "4":
            show_formats()
        elif choice == "5":
            quick_examples(predictor)
        elif choice == "6":
            print("Thank you for using the Disease Prediction System!")
            break
        else:
            print("Invalid choice. Please enter a number between 1-6.")

def heart_disease_demo(predictor):
    """Interactive heart disease prediction"""
    print("\nHEART DISEASE PREDICTION DEMO")
    print("-" * 40)
    
    # Pre-defined examples
    examples = {
        "1": {
            "data": [35, 0, 0, 110, 180, 0, 0, 170, 0, 0.0, 1, 0, 3],
            "description": "Healthy 35-year-old female, good vitals"
        },
        "2": {
            "data": [65, 1, 3, 160, 286, 1, 2, 108, 1, 2.0, 2, 3, 6],
            "description": "65-year-old male, high BP & cholesterol, chest pain"
        },
        "3": {
            "data": [45, 1, 2, 140, 230, 0, 1, 150, 0, 1.2, 1, 1, 3],
            "description": "45-year-old male, moderate risk factors"
        }
    }
    
    print("Choose a patient example or enter custom data:")
    print("1. Low Risk Patient")
    print("2. High Risk Patient") 
    print("3. Medium Risk Patient")
    print("4. Enter Custom Data")
    print("5. Back to Main Menu")
    
    choice = input("Choice (1-5): ").strip()
    
    if choice in ["1", "2", "3"]:
        example = examples[choice]
        print(f"\nUsing: {example['description']}")
        print(f"Input Data: {example['data']}")
        
        # Get best heart disease model
        if 'heart_disease' in predictor.models:
            best_model = max(predictor.models['heart_disease'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('heart_disease', best_model, example['data'])
            print_result(result, "Heart Disease")
        else:
            print("Heart disease model not available")
            
    elif choice == "4":
        custom_heart_input(predictor)
    elif choice == "5":
        return
    else:
        print("Invalid choice")

def custom_heart_input(predictor):
    """Get custom heart disease input from user"""
    print("\nCUSTOM HEART DISEASE INPUT")
    print("-" * 35)
    print("Please enter the following information:")
    
    try:
        age = int(input("Age (20-80): "))
        print("Sex: 0=Female, 1=Male")
        sex = int(input("Sex (0 or 1): "))
        print("Chest Pain: 0=None, 1=Typical, 2=Atypical, 3=Non-cardiac")
        cp = int(input("Chest Pain Type (0-3): "))
        trestbps = int(input("Resting Blood Pressure (90-200): "))
        chol = int(input("Cholesterol Level (100-400): "))
        print("Fasting Blood Sugar > 120: 0=No, 1=Yes")
        fbs = int(input("High Fasting Blood Sugar (0 or 1): "))
        
        # Use reasonable defaults for complex medical parameters
        restecg = 1  # Normal ECG
        thalach = 150  # Average max heart rate
        exang = 0  # No exercise angina
        oldpeak = 1.0  # Mild ST depression
        slope = 1  # Flat slope
        ca = 0  # No major vessels
        thal = 3  # Normal thalassemia
        
        custom_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        print(f"\nYour Input: {custom_data}")
        
        if 'heart_disease' in predictor.models:
            best_model = max(predictor.models['heart_disease'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('heart_disease', best_model, custom_data)
            print_result(result, "Heart Disease")
            
            # Additional recommendations
            if isinstance(result, dict) and result['prediction'] == 1:
                print(f"\nIMPORTANT RECOMMENDATIONS:")
                print(f"   - Consult a cardiologist immediately")
                print(f"   - Monitor blood pressure regularly")
                print(f"   - Consider lifestyle changes (diet, exercise)")
                print(f"   - This is not a medical diagnosis - see a doctor!")
                
        else:
            print("Heart disease model not available")
            
    except ValueError:
        print("Invalid input. Please enter numbers only.")
    except Exception as e:
        print(f"Error: {e}")

def diabetes_demo(predictor):
    """Interactive diabetes prediction"""
    print("\nDIABETES PREDICTION DEMO")
    print("-" * 35)
    
    examples = {
        "1": {
            "data": [1, 85, 66, 29, 94, 26.6, 0.351, 25],
            "description": "Healthy 25-year-old, normal glucose & BMI"
        },
        "2": {
            "data": [8, 183, 64, 0, 0, 23.3, 0.672, 45],
            "description": "45-year-old, 8 pregnancies, very high glucose"
        },
        "3": {
            "data": [3, 120, 78, 30, 100, 32.5, 0.4, 35],
            "description": "35-year-old, moderate risk factors"
        }
    }
    
    print("Choose a patient example:")
    print("1. Low Risk Patient")
    print("2. High Risk Patient")
    print("3. Medium Risk Patient")
    print("4. Custom Input")
    print("5. Back")
    
    choice = input("Choice (1-5): ").strip()
    
    if choice in ["1", "2", "3"]:
        example = examples[choice]
        print(f"\nUsing: {example['description']}")
        
        if 'diabetes' in predictor.models:
            best_model = max(predictor.models['diabetes'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('diabetes', best_model, example['data'])
            print_result(result, "Diabetes")
        else:
            print("Diabetes model not available")
            
    elif choice == "4":
        custom_diabetes_input(predictor)
    elif choice == "5":
        return

def custom_diabetes_input(predictor):
    """Get custom diabetes input from user"""
    print("\nCUSTOM DIABETES INPUT")
    print("-" * 30)
    
    try:
        pregnancies = int(input("Number of pregnancies (0-15): "))
        glucose = int(input("Glucose level (50-200): "))
        bp = int(input("Blood pressure (50-120): "))
        skin = int(input("Skin thickness mm (0-60, 0 if unknown): "))
        insulin = int(input("Insulin level (0-800, 0 if unknown): "))
        bmi = float(input("BMI (15-50): "))
        pedigree = float(input("Diabetes pedigree function (0.1-2.0): "))
        age = int(input("Age (18-80): "))
        
        custom_data = [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
        
        print(f"\nYour Input: {custom_data}")
        
        if 'diabetes' in predictor.models:
            best_model = max(predictor.models['diabetes'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('diabetes', best_model, custom_data)
            print_result(result, "Diabetes")
        else:
            print("Diabetes model not available")
            
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"Error: {e}")

def breast_cancer_demo(predictor):
    """Breast cancer prediction demo"""
    print("\nBREAST CANCER PREDICTION DEMO")
    print("-" * 40)
    
    print("Breast cancer prediction uses 30 numerical features.")
    print("For demo purposes, we'll use pre-defined examples:")
    print("1. Benign (Low Risk) Sample")
    print("2. Malignant (High Risk) Sample")
    print("3. Back to Main Menu")
    
    choice = input("Choice (1-3): ").strip()
    
    if choice == "1":
        # Benign example
        benign_data = [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744,
                      0.4956, 1.156, 3.445, 27.23, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208,
                      14.91, 26.5, 98.87, 567.7, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173]
        
        print("Using: Small tumor, regular features (likely benign)")
        
        if 'breast_cancer' in predictor.models:
            best_model = max(predictor.models['breast_cancer'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('breast_cancer', best_model, benign_data)
            print_result(result, "Breast Cancer")
        else:
            print("Breast cancer model not available")
            
    elif choice == "2":
        # Malignant example
        malignant_data = [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
                         0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
                         24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902]
        
        print("Using: Large tumor, irregular features (concerning)")
        
        if 'breast_cancer' in predictor.models:
            best_model = max(predictor.models['breast_cancer'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
            
            result = predictor.predict_new_patient('breast_cancer', best_model, malignant_data)
            print_result(result, "Breast Cancer")
        else:
            print("Breast cancer model not available")
            
    elif choice == "3":
        return

def print_result(result, disease_type):
    """Print prediction result in a formatted way"""
    print(f"\n{disease_type.upper()} PREDICTION RESULT:")
    print("=" * 45)
    
    if isinstance(result, dict):
        print(f"   Risk Assessment: {result['risk_level']}")
        if result['confidence']:
            print(f"   Confidence Level: {result['confidence']:.1%}")
        print(f"   Model Used: {result['model_used']}")
        
        # Risk-specific recommendations
        if result['prediction'] == 1:
            print(f"\nHIGH RISK DETECTED!")
            print(f"   RECOMMENDATIONS:")
            if disease_type == "Heart Disease":
                print(f"   - Consult a cardiologist immediately")
                print(f"   - Monitor blood pressure and cholesterol")
                print(f"   - Adopt heart-healthy lifestyle changes")
            elif disease_type == "Diabetes":
                print(f"   - Consult an endocrinologist")
                print(f"   - Monitor blood glucose levels")
                print(f"   - Consider dietary modifications")
            elif disease_type == "Breast Cancer":
                print(f"   - Consult an oncologist immediately")
                print(f"   - Schedule further diagnostic tests")
                print(f"   - Consider additional imaging studies")
            
            print(f"   WARNING: This is NOT a medical diagnosis!")
            print(f"   WARNING: Please see a healthcare professional!")
            
        else:
            print(f"\nLOW RISK INDICATED")
            print(f"   RECOMMENDATIONS:")
            print(f"   - Continue maintaining healthy lifestyle")
            print(f"   - Regular health check-ups")
            print(f"   - Stay aware of risk factors")
            
    else:
        print(f"Prediction Error: {result}")

def show_formats():
    """Display input format guide"""
    print(f"\nINPUT FORMAT REFERENCE GUIDE")
    print("=" * 45)
    
    print(f"\nHEART DISEASE INPUT (13 values):")
    print(f"   [age, sex, chest_pain_type, resting_bp, cholesterol,")
    print(f"    fasting_sugar, resting_ecg, max_heart_rate, exercise_angina,") 
    print(f"    st_depression, slope, major_vessels, thalassemia]")
    print(f"   Example: [65, 1, 0, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3]")
    
    print(f"\nDIABETES INPUT (8 values):")
    print(f"   [pregnancies, glucose, blood_pressure, skin_thickness,")
    print(f"    insulin, bmi, diabetes_pedigree_function, age]")
    print(f"   Example: [6, 148, 72, 35, 0, 33.6, 0.627, 50]")
    
    print(f"\nBREAST CANCER INPUT (30 values):")
    print(f"   [mean_radius, mean_texture, mean_perimeter, mean_area,")
    print(f"    mean_smoothness, ... 25 more numerical features]")
    print(f"   Example: [17.99, 10.38, 122.8, 1001.0, 0.1184, ...]")
    print(f"   TIP: Use pre-defined examples for breast cancer demo")

def quick_examples(predictor):
    """Run quick prediction examples"""
    print(f"\nQUICK PREDICTION EXAMPLES")
    print("=" * 35)
    
    print("Running predictions on sample data...\n")
    
    # Heart Disease Example
    if 'heart_disease' in predictor.models:
        heart_data = [65, 1, 0, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3]
        best_heart_model = max(predictor.models['heart_disease'].items(), 
                              key=lambda x: x[1]['accuracy'])[0]
        heart_result = predictor.predict_new_patient('heart_disease', best_heart_model, heart_data)
        if isinstance(heart_result, dict):
            print(f"Heart Disease: {heart_result['risk_level']} (Confidence: {heart_result['confidence']:.1%})")
    
    # Diabetes Example
    if 'diabetes' in predictor.models:
        diabetes_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        best_diabetes_model = max(predictor.models['diabetes'].items(), 
                                 key=lambda x: x[1]['accuracy'])[0]
        diabetes_result = predictor.predict_new_patient('diabetes', best_diabetes_model, diabetes_data)
        if isinstance(diabetes_result, dict):
            print(f"Diabetes: {diabetes_result['risk_level']} (Confidence: {diabetes_result['confidence']:.1%})")
    
    # Breast Cancer Example
    if 'breast_cancer' in predictor.models:
        bc_data = [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
                   0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
                   24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902]
        best_bc_model = max(predictor.models['breast_cancer'].items(), 
                           key=lambda x: x[1]['accuracy'])[0]
        bc_result = predictor.predict_new_patient('breast_cancer', best_bc_model, bc_data)
        if isinstance(bc_result, dict):
            print(f"Breast Cancer: {bc_result['risk_level']} (Confidence: {bc_result['confidence']:.1%})")
    
    print(f"\nQuick examples completed!")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Make sure task4.py is in the same directory and working properly.")
