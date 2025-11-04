"""
Verification Script for Bank Customer Churn Prediction System
Run this script to verify that everything is set up correctly before training models.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_status(test_name, passed, message=""):
    """Print test status"""
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {test_name}")
    if message:
        print(f"      {message}")

def verify_python_version():
    """Check Python version"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_status(
        "Python Version",
        is_valid,
        f"Found Python {version_str} (Required: 3.8+)"
    )
    return is_valid

def verify_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required_packages = {
        'streamlit': '1.28.0',
        'pandas': '2.0.3',
        'numpy': '1.24.3',
        'sklearn': '1.3.0',
        'xgboost': '2.0.0',
        'tensorflow': '2.13.0',
        'matplotlib': '3.7.2',
        'seaborn': '0.12.2',
        'plotly': '5.16.1',
        'imblearn': '0.11.0',
        'mlxtend': '0.23.0',
        'joblib': '1.3.2'
    }

    all_installed = True

    for package, expected_version in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'imblearn':
                import imblearn
                version = imblearn.__version__
            else:
                module = __import__(package)
                version = module.__version__

            print_status(f"{package}", True, f"Version {version}")
        except ImportError:
            print_status(f"{package}", False, f"NOT INSTALLED (Required: {expected_version})")
            all_installed = False

    return all_installed

def verify_project_structure():
    """Check if project structure is correct"""
    print_header("Checking Project Structure")

    required_files = [
        'app.py',
        'train_models.py',
        'utils.py',
        'requirements.txt',
        'README.md'
    ]

    required_dirs = [
        'data',
        'models',
        'results'
    ]

    all_exists = True

    for file in required_files:
        exists = Path(file).exists()
        print_status(f"File: {file}", exists)
        all_exists = all_exists and exists

    for dir in required_dirs:
        exists = Path(dir).exists()
        print_status(f"Directory: {dir}/", exists)
        all_exists = all_exists and exists

    return all_exists

def verify_dataset():
    """Check if dataset exists"""
    print_header("Checking Dataset")

    dataset_path = Path('data/Churn_Modelling.csv')

    if not dataset_path.exists():
        print_status("Dataset", False, "data/Churn_Modelling.csv NOT FOUND")
        print("\n" + "!"*80)
        print("  IMPORTANT: Dataset is missing!")
        print("  Please download from: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction")
        print("  Place Churn_Modelling.csv in the data/ directory")
        print("!"*80)
        return False

    # Try to read dataset
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)

        expected_rows_min = 9900  # Allow some flexibility
        expected_rows_max = 10100
        expected_cols = 14

        if expected_rows_min <= df.shape[0] <= expected_rows_max and df.shape[1] == expected_cols:
            print_status("Dataset", True, f"Found {df.shape[0]} rows, {df.shape[1]} columns (Expected: ~10,000)")

            # Check for required columns
            required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                              'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                              'EstimatedSalary', 'Exited']

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print_status("Dataset Columns", False, f"Missing columns: {missing_columns}")
                return False
            else:
                print_status("Dataset Columns", True, "All required columns present")
                return True
        else:
            print_status("Dataset", False,
                        f"Unexpected size: {df.shape[0]} rows, {df.shape[1]} columns (Expected: ~10,000 rows x {expected_cols} columns)")
            return False

    except Exception as e:
        print_status("Dataset", False, f"Error reading dataset: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print_header("Bank Customer Churn Prediction System - Setup Verification")
    print("This script checks if your environment is ready for model training.")

    # Check Python version
    print_header("Checking Python Environment")
    python_ok = verify_python_version()

    # Check dependencies
    dependencies_ok = verify_dependencies()

    # Check project structure
    structure_ok = verify_project_structure()

    # Check dataset
    dataset_ok = verify_dataset()

    # Final summary
    print_header("Verification Summary")

    all_checks = [
        ("Python Version", python_ok),
        ("Dependencies", dependencies_ok),
        ("Project Structure", structure_ok),
        ("Dataset", dataset_ok)
    ]

    passed = sum(1 for _, status in all_checks if status)
    total = len(all_checks)

    for check_name, status in all_checks:
        print_status(check_name, status)

    print("\n" + "-"*80)
    print(f"Overall: {passed}/{total} checks passed")
    print("-"*80)

    if passed == total:
        print("\n" + "="*80)
        print("  SUCCESS! Your environment is ready.")
        print("  You can now run: python train_models.py")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("  SETUP INCOMPLETE")
        print("  Please fix the issues above before proceeding.")
        print("="*80)

        if not dependencies_ok:
            print("\n  To install dependencies, run:")
            print("    pip install -r requirements.txt")

        if not dataset_ok:
            print("\n  To get the dataset:")
            print("    1. Go to: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction")
            print("    2. Download Churn_Modelling.csv")
            print("    3. Place it in the data/ directory")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
