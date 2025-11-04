# Quick Start Guide

Get up and running with the Bank Customer Churn Prediction System in 5 minutes!

---

## Step 1: Install Dependencies (2 minutes)

Open terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This installs all required packages:
- streamlit
- pandas, numpy
- scikit-learn, xgboost, tensorflow
- matplotlib, seaborn, plotly
- imbalanced-learn, mlxtend

---

## Step 2: Download Dataset (1 minute)

1. Go to: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
2. Click "Download" button (you may need to sign in to Kaggle)
3. Extract `Churn_Modelling.csv` from the ZIP file
4. Place it in the `data/` folder

**Verify**: You should have `data/Churn_Modelling.csv`

---

## Step 3: Train Models (15-30 minutes)

Run the training pipeline:

```bash
python train_models.py
```

**What happens**:
- Loads and analyzes dataset
- Preprocesses data (scaling, encoding, SMOTE)
- Trains 7 ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Gradient Boosting, Neural Network)
- Performs Grid Search CV on Random Forest and XGBoost
- Performs K-Means clustering
- Performs Association Rule Mining
- Evaluates all models
- Saves models to `models/` directory
- Generates visualizations in `results/` directory

**Expected output**: Console logs showing progress and metrics

**Time**: 15-30 minutes depending on your computer

**Grab a coffee while it runs!**

---

## Step 4: Launch Streamlit App (30 seconds)

After training completes, run:

```bash
streamlit run app.py
```

**What happens**:
- Streamlit server starts
- Browser opens automatically to http://localhost:8501
- Professional dashboard loads with 5 pages

---

## Step 5: Explore the Dashboard

### Page 1: Home
- Read about the business problem
- View dataset overview
- Understand the 5 Course Outcomes

### Page 2: Predict Churn
- Enter customer information using the form
- Select a model
- Click "Predict Churn Risk"
- View churn probability, risk level, and personalized recommendations

### Page 3: Data Analytics
- Explore 4 tabs:
  - Dataset Overview
  - Exploratory Analysis (charts and insights)
  - Customer Segmentation (clustering)
  - Churn Patterns (association rules)

### Page 4: Model Performance
- Compare all 7 models
- View ROC curves and confusion matrices
- Understand business cost analysis
- See model recommendations

### Page 5: Batch Predictions
- Upload a CSV file with customer data
- Predict churn for all customers
- View risk stratification
- Calculate retention campaign ROI
- Download results

---

## Quick Test

Want to test immediately? Use these sample customer values in Page 2:

**High Risk Customer**:
- Geography: Germany
- Gender: Female
- Age: 55
- Credit Score: 600
- Tenure: 2 years
- Balance: $125,000
- Number of Products: 1
- Has Credit Card: No
- Is Active Member: No
- Estimated Salary: $80,000

**Expected**: HIGH RISK (70-80% churn probability)

**Low Risk Customer**:
- Geography: France
- Gender: Male
- Age: 35
- Credit Score: 750
- Tenure: 8 years
- Balance: $100,000
- Number of Products: 2
- Has Credit Card: Yes
- Is Active Member: Yes
- Estimated Salary: $120,000

**Expected**: LOW RISK (10-20% churn probability)

---

## Troubleshooting

**Problem**: "ModuleNotFoundError"
- **Solution**: Run `pip install -r requirements.txt`

**Problem**: "FileNotFoundError: data/Churn_Modelling.csv"
- **Solution**: Download dataset from Kaggle and place in `data/` folder

**Problem**: "No models found"
- **Solution**: Run `python train_models.py` first

**Problem**: Training is slow
- **Solution**: This is normal. The process trains 7 models + Grid Search. Takes 15-30 minutes.

**Problem**: Streamlit won't start
- **Solution**: Make sure training completed successfully and models are saved in `models/` directory

---

## Project Structure

```
bank-churn-prediction/
├── app.py                  # Streamlit dashboard
├── train_models.py         # Model training pipeline
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── README.md              # Full documentation
├── QUICK_START.md         # This file
│
├── data/
│   └── Churn_Modelling.csv   # Dataset (you download)
│
├── models/                    # Trained models (created by train_models.py)
│   ├── *.pkl                  # Scikit-learn models
│   └── neural_network.h5      # TensorFlow model
│
└── results/                   # Generated plots (created by train_models.py)
    ├── *.html                 # Interactive Plotly charts
    └── *.csv                  # Results tables
```

---

## Next Steps

1. **Read Full Documentation**: Check [README.md](README.md) for detailed explanations
2. **Understand Course Outcomes**: See how each CO is demonstrated
3. **Experiment**: Try different customer profiles in the prediction page
4. **Batch Processing**: Upload your own customer dataset
5. **Customize**: Modify code to add new features or models

---

## Support

Having issues? Check:
1. [README.md](README.md) - Full documentation
2. Console output - Error messages provide hints
3. Requirements - Ensure all packages installed correctly
4. Dataset - Verify CSV file is in correct location

---

## Congratulations!

You now have a fully functional ML-powered churn prediction system running on your local machine!

Explore all 5 pages to see the comprehensive features.

**Enjoy predicting churn and saving customers!**
