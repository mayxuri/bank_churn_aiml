"""
Bank Customer Churn Prediction - Model Training Pipeline
Complete ML system covering 5 Course Outcomes (COs)

CO1: AI-based Heuristic Techniques - Grid Search CV for hyperparameter optimization
CO2: Data Preprocessing - Scaling, encoding, SMOTE, feature engineering
CO3: Supervised Learning - Multiple classification models with comprehensive evaluation
CO4: Unsupervised Learning - K-Means clustering and Association Rule Mining
CO5: Neural Networks - Deep learning with TensorFlow/Keras

"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import time
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.cluster import KMeans

# Imbalanced-learn for SMOTE
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# TensorFlow/Keras for Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# Association rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Custom utilities
from utils import (
    discretize_features, engineer_features,
    calculate_business_cost, interpret_cluster
)

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BankChurnModelTrainer:
    """
    Comprehensive model training pipeline for bank customer churn prediction.
    """

    def __init__(self, data_path='data/Churn_Modelling.csv'):
        # Purpose: Initialize the training pipeline object and allocate placeholders for
        #          data, preprocessing artifacts, trained models and results.
        # What: Constructor for BankChurnModelTrainer. It sets default file paths and
        #       internal attributes used across the pipeline (dataframes, scalers,
        #       encoder placeholders, model/result stores).
        # Why: Centralizes configuration so other methods can rely on pre-initialized
        #      attributes and simplifies running the full pipeline from a single entry.
        # Params:
        #   - data_path (str): path to input CSV. Default 'data/Churn_Modelling.csv'.
        #     This allows the pipeline to locate and load the training dataset.
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.label_encoder_gender = None
        self.feature_names = None

        self.models = {}
        self.results = {}
        self.training_times = {}

        print("="*80)
        print("BANK CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
        print("="*80)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Random State: {RANDOM_STATE}")
        print("="*80)

    def load_and_explore_data(self):
        # Purpose: Load the raw dataset and perform initial exploratory analysis.
        # What: Reads CSV into a pandas DataFrame, prints basic summaries, checks
        #       for missing values, and computes target distribution statistics.
        # Why: Understand data quality and target imbalance before preprocessing.
        # Params: None. Uses self.data_path provided during initialization.
        print("\n[STEP 1] LOADING AND EXPLORING DATA")
        print("-" * 80)

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully from: {self.data_path}")
        print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        # Display basic info
        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

        # Check for missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values detected!")
        else:
            print(missing[missing > 0])

        # Target variable distribution
        print("\nTarget Variable Distribution (Exited):")
        churn_counts = self.df['Exited'].value_counts()
        churn_pct = self.df['Exited'].value_counts(normalize=True) * 100
        print(f"  Not Churned (0): {churn_counts[0]} ({churn_pct[0]:.2f}%)")
        print(f"  Churned (1): {churn_counts[1]} ({churn_pct[1]:.2f}%)")
        print(f"  Churn Rate: {churn_pct[1]:.2f}%")
        print(f"  Class Imbalance Ratio: {churn_counts[0]/churn_counts[1]:.2f}:1")

        # Visualizations
        self._create_eda_plots()

        print("\n[STEP 1] COMPLETED: Data loaded and explored successfully")

    def _create_eda_plots(self):
        # Purpose: Produce and persist exploratory visualizations for reporting.
        # What: Generates pie charts, bar charts, histograms and a correlation heatmap
        #       (saved as interactive HTML files in results/).
        # Why: Visual aids help identify feature distributions, correlations and
        #      demographic patterns relevant to churn.
        # Params: None. Reads from self.df which must be set by load_and_explore_data().
        print("\nGenerating EDA visualizations...")

        # 1. Target distribution
        fig = px.pie(
            self.df,
            names='Exited',
            title='Customer Churn Distribution',
            color='Exited',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        fig.write_html('results/target_distribution.html')

        # 2. Churn by Geography
        geo_churn = self.df.groupby('Geography')['Exited'].agg(['sum', 'count'])
        geo_churn['churn_rate'] = (geo_churn['sum'] / geo_churn['count'] * 100)
        fig = px.bar(
            geo_churn,
            y='churn_rate',
            title='Churn Rate by Geography',
            labels={'churn_rate': 'Churn Rate (%)', 'Geography': 'Country'},
            color='churn_rate',
            color_continuous_scale='Reds'
        )
        fig.write_html('results/churn_by_geography.html')

        # 3. Churn by Gender
        gender_churn = self.df.groupby('Gender')['Exited'].agg(['sum', 'count'])
        gender_churn['churn_rate'] = (gender_churn['sum'] / gender_churn['count'] * 100)
        fig = px.bar(
            gender_churn,
            y='churn_rate',
            title='Churn Rate by Gender',
            labels={'churn_rate': 'Churn Rate (%)'},
            color='churn_rate',
            color_continuous_scale='Oranges'
        )
        fig.write_html('results/churn_by_gender.html')

        # 4. Age distribution
        fig = px.histogram(
            self.df,
            x='Age',
            color='Exited',
            title='Age Distribution by Churn Status',
            nbins=30,
            barmode='overlay',
            opacity=0.7
        )
        fig.write_html('results/age_distribution.html')

        # 5. Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0
        ))
        fig.update_layout(title='Feature Correlation Heatmap', height=800, width=900)
        fig.write_html('results/correlation_heatmap.html')

        print("EDA plots saved to results/ directory")

    def preprocess_data(self):
        # Purpose: Prepare raw data into a modeling-ready format.
        # What: Handles missing values, performs feature engineering (ratios and
        #       categorical binning), encodes categorical variables, splits into
        #       train/test sets, scales numeric features and applies SMOTE to
        #       balance the training set.
        # Why: Standardization of features, encoding and class balancing are
        #      required for reliable training across multiple model types.
        # Params: None (operates on self.df). Produces attributes used later:
        #   - self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test,
        #   - self.scaler, self.feature_names, self.label_encoder_gender
        print("\n[STEP 2] DATA PREPROCESSING")
        print("-" * 80)

        # Step 2.0: Handle Missing Values
        print("\n[2.0] Handling Missing Values")
        print(f"Missing values before handling:")
        missing_before = self.df.isnull().sum()
        print(missing_before[missing_before > 0])

        # Fill missing values with appropriate strategies
        # Geography: fill with mode
        if self.df['Geography'].isnull().sum() > 0:
            self.df['Geography'].fillna(self.df['Geography'].mode()[0], inplace=True)

        # Age: fill with median
        if self.df['Age'].isnull().sum() > 0:
            self.df['Age'].fillna(self.df['Age'].median(), inplace=True)

        # HasCrCard: fill with mode (most common value)
        if self.df['HasCrCard'].isnull().sum() > 0:
            self.df['HasCrCard'].fillna(self.df['HasCrCard'].mode()[0], inplace=True)

        # IsActiveMember: fill with mode
        if self.df['IsActiveMember'].isnull().sum() > 0:
            self.df['IsActiveMember'].fillna(self.df['IsActiveMember'].mode()[0], inplace=True)

        print(f"\nMissing values after handling:")
        missing_after = self.df.isnull().sum()
        if missing_after.sum() == 0:
            print("All missing values have been handled!")
        else:
            print(missing_after[missing_after > 0])

        # Step 2.1: Feature Engineering
        print("\n[2.1] Feature Engineering")
        self.df_processed = engineer_features(self.df)
        print("Created engineered features:")
        print("  - BalanceSalaryRatio = Balance / EstimatedSalary")
        print("  - TenureAgeRatio = Tenure / Age")
        print("  - BalancePerProduct = Balance / NumOfProducts")
        print("  - AgeGroup: Young(<35), Middle(35-50), Senior(>50)")
        print("  - BalanceCategory: Low(<50k), Medium(50k-100k), High(>100k)")
        print("  - CreditScoreCategory: Poor(<600), Fair(600-700), Good(>700)")

        # Step 2.2: Drop irrelevant columns
        print("\n[2.2] Dropping non-predictive columns")
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        self.df_processed = self.df_processed.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")

        # Step 2.3: Encode categorical variables
        print("\n[2.3] Encoding Categorical Variables")

        # Gender: Label Encoding (Male=1, Female=0)
        self.label_encoder_gender = LabelEncoder()
        self.df_processed['Gender'] = self.label_encoder_gender.fit_transform(
            self.df_processed['Gender']
        )
        print(f"  Gender encoded: {dict(zip(self.label_encoder_gender.classes_, self.label_encoder_gender.transform(self.label_encoder_gender.classes_)))}")

        # Geography: One-Hot Encoding
        geography_dummies = pd.get_dummies(
            self.df_processed['Geography'],
            prefix='Geography',
            drop_first=False
        )
        self.df_processed = pd.concat([self.df_processed, geography_dummies], axis=1)
        self.df_processed = self.df_processed.drop('Geography', axis=1)
        print(f"  Geography one-hot encoded: {list(geography_dummies.columns)}")

        # Encode other categorical features created during feature engineering
        categorical_features = ['AgeGroup', 'BalanceCategory', 'CreditScoreCategory']
        for col in categorical_features:
            if col in self.df_processed.columns:
                dummies = pd.get_dummies(self.df_processed[col], prefix=col, drop_first=True)
                self.df_processed = pd.concat([self.df_processed, dummies], axis=1)
                self.df_processed = self.df_processed.drop(col, axis=1)
        print(f"  Encoded engineered categorical features: {categorical_features}")

        # Step 2.4: Split features and target
        print("\n[2.4] Splitting Features and Target")
        X = self.df_processed.drop('Exited', axis=1)
        y = self.df_processed['Exited']

        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"Total features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")

        # Step 2.5: Train-Test Split
        print("\n[2.5] Train-Test Split (80-20)")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y  # Maintain class distribution
        )
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Training set churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"Test set churn rate: {self.y_test.mean()*100:.2f}%")

        # Step 2.6: Feature Scaling
        print("\n[2.6] Feature Scaling (StandardScaler)")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Features scaled to zero mean and unit variance")

        # Step 2.7: Handle Class Imbalance with SMOTE
        print("\n[2.7] Handling Class Imbalance with SMOTE")
        print(f"Before SMOTE - Class distribution:")
        print(f"  Class 0 (Not Churned): {(self.y_train == 0).sum()}")
        print(f"  Class 1 (Churned): {(self.y_train == 1).sum()}")

        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
        self.X_train_scaled, self.y_train = smote.fit_resample(
            self.X_train_scaled,
            self.y_train
        )

        print(f"After SMOTE - Class distribution:")
        print(f"  Class 0 (Not Churned): {(self.y_train == 0).sum()}")
        print(f"  Class 1 (Churned): {(self.y_train == 1).sum()}")
        print("Classes are now balanced!")

        print("\n[STEP 2] COMPLETED: Data preprocessing finished")

    def train_traditional_models(self):
        # Purpose: Train a suite of traditional supervised classifiers and evaluate them.
        # What: Fits Logistic Regression, Decision Tree, Random Forest, XGBoost,
        #       SVM and Gradient Boosting on the preprocessed training data, then
        #       evaluates on the test set to compute standard metrics.
        # Why: Provides baseline and ensemble comparisons; different algorithms
        #      capture different signal patterns (linear vs tree-based vs kernel).
        # Params: Uses self.X_train_scaled, self.y_train and self.X_test_scaled.
        #       Model hyperparameters are chosen for reasonable defaults to
        #       balance performance and training time.
        print("\n[STEP 3] TRAINING TRADITIONAL ML MODELS")
        print("-" * 80)

        # Define models
        models_to_train = {
            'Logistic Regression': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            ),
            'SVM': SVC(
                random_state=RANDOM_STATE,
                kernel='rbf',
                probability=True,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
        }

        # Train each model
        for name, model in models_to_train.items():
            print(f"\n[3.{list(models_to_train.keys()).index(name)+1}] Training {name}...")
            start_time = time.time()

            # Train
            model.fit(self.X_train_scaled, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # Store results
            training_time = time.time() - start_time
            self.models[name] = model
            self.results[name] = metrics
            self.training_times[name] = training_time

            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        print("\n[STEP 3] COMPLETED: Traditional models trained successfully")

    def hyperparameter_tuning(self):
        # Purpose: Improve model performance by searching for better hyperparameters.
        # What: Runs GridSearchCV for Random Forest and XGBoost over predefined
        #       parameter grids using 5-fold CV and optimizing recall (priority
        #       is to catch churners).
        # Why: Tuning boosts recall and overall robustness; recall is prioritized
        #      because false negatives (missed churners) are costly to the business.
        # Params: Parameter grids defined inside method (n_estimators, max_depth,
        #       learning_rate, subsample, etc.). Uses self.X_train_scaled/self.y_train.
        print("\n[STEP 4] HYPERPARAMETER OPTIMIZATION (GRID SEARCH CV)")
        print("-" * 80)
        print("CO1: AI-based Heuristic Techniques")
        print("Using Grid Search CV to find optimal hyperparameters")

        # Random Forest Grid Search
        print("\n[4.1] Optimizing Random Forest")
        print("Default Random Forest Performance:")
        rf_default = self.results['Random Forest']
        print(f"  Accuracy: {rf_default['accuracy']:.4f}")
        print(f"  Recall: {rf_default['recall']:.4f}")
        print(f"  ROC-AUC: {rf_default['roc_auc']:.4f}")

        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6]
        }

        print(f"\nGrid Search parameters: {rf_param_grid}")
        print("Performing 5-fold cross-validation...")

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
            rf_param_grid,
            cv=5,
            scoring='recall',  # Prioritize recall for churn prediction
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        rf_grid.fit(self.X_train_scaled, self.y_train)
        rf_time = time.time() - start_time

        print(f"\nGrid Search completed in {rf_time:.2f}s")
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Best cross-validation score (Recall): {rf_grid.best_score_:.4f}")

        # Evaluate optimized model
        rf_optimized = rf_grid.best_estimator_
        y_pred = rf_optimized.predict(self.X_test_scaled)
        y_pred_proba = rf_optimized.predict_proba(self.X_test_scaled)[:, 1]
        rf_opt_metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

        print("\nOptimized Random Forest Performance:")
        print(f"  Accuracy: {rf_opt_metrics['accuracy']:.4f} (Δ: {rf_opt_metrics['accuracy']-rf_default['accuracy']:+.4f})")
        print(f"  Recall: {rf_opt_metrics['recall']:.4f} (Δ: {rf_opt_metrics['recall']-rf_default['recall']:+.4f})")
        print(f"  ROC-AUC: {rf_opt_metrics['roc_auc']:.4f} (Δ: {rf_opt_metrics['roc_auc']-rf_default['roc_auc']:+.4f})")

        self.models['Random Forest Optimized'] = rf_optimized
        self.results['Random Forest Optimized'] = rf_opt_metrics
        self.training_times['Random Forest Optimized'] = rf_time

        # XGBoost Grid Search
        print("\n[4.2] Optimizing XGBoost")
        print("Default XGBoost Performance:")
        xgb_default = self.results['XGBoost']
        print(f"  Accuracy: {xgb_default['accuracy']:.4f}")
        print(f"  Recall: {xgb_default['recall']:.4f}")
        print(f"  ROC-AUC: {xgb_default['roc_auc']:.4f}")

        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }

        print(f"\nGrid Search parameters: {xgb_param_grid}")
        print("Performing 5-fold cross-validation...")

        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            xgb_param_grid,
            cv=5,
            scoring='recall',
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        xgb_grid.fit(self.X_train_scaled, self.y_train)
        xgb_time = time.time() - start_time

        print(f"\nGrid Search completed in {xgb_time:.2f}s")
        print(f"Best parameters: {xgb_grid.best_params_}")
        print(f"Best cross-validation score (Recall): {xgb_grid.best_score_:.4f}")

        # Evaluate optimized model
        xgb_optimized = xgb_grid.best_estimator_
        y_pred = xgb_optimized.predict(self.X_test_scaled)
        y_pred_proba = xgb_optimized.predict_proba(self.X_test_scaled)[:, 1]
        xgb_opt_metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

        print("\nOptimized XGBoost Performance:")
        print(f"  Accuracy: {xgb_opt_metrics['accuracy']:.4f} (Δ: {xgb_opt_metrics['accuracy']-xgb_default['accuracy']:+.4f})")
        print(f"  Recall: {xgb_opt_metrics['recall']:.4f} (Δ: {xgb_opt_metrics['recall']-xgb_default['recall']:+.4f})")
        print(f"  ROC-AUC: {xgb_opt_metrics['roc_auc']:.4f} (Δ: {xgb_opt_metrics['roc_auc']-xgb_default['roc_auc']:+.4f})")

        self.models['XGBoost Optimized'] = xgb_optimized
        self.results['XGBoost Optimized'] = xgb_opt_metrics
        self.training_times['XGBoost Optimized'] = xgb_time

        print("\n[STEP 4] COMPLETED: Hyperparameter optimization finished")
        print("Grid Search demonstrates CO1: AI-based heuristic search improves model performance")

    def train_neural_network(self):
        # Purpose: Train a deep neural network to capture complex non-linear interactions.
        # What: Builds a Sequential Keras model (Dense layers with BatchNorm + Dropout)
        #       and trains it with Adam optimizer, binary cross-entropy loss and AUC metric.
        # Why: Neural networks can model higher-order feature interactions that
        #      tree-based models may not capture; used as an additional model type.
        # Params:
        #   - architecture: [128, 64, 32] hidden units with ReLU
        #   - regularization: Dropout (0.3/0.2) and BatchNormalization
        #   - training: epochs=100, batch_size=32, validation_split=0.2,
        #     early stopping (patience=15), checkpoint saving to models/neural_network.h5
        print("\n[STEP 5] TRAINING NEURAL NETWORK")
        print("-" * 80)
        print("CO5: Neural Networks - Deep Learning with TensorFlow/Keras")

        # Build neural network architecture
        print("\n[5.1] Building Neural Network Architecture")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        print("Architecture:")
        print("  Input Layer: {} features".format(self.X_train_scaled.shape[1]))
        print("  Hidden Layer 1: 128 neurons (ReLU) + BatchNorm + Dropout(0.3)")
        print("  Hidden Layer 2: 64 neurons (ReLU) + BatchNorm + Dropout(0.3)")
        print("  Hidden Layer 3: 32 neurons (ReLU) + BatchNorm + Dropout(0.2)")
        print("  Output Layer: 1 neuron (Sigmoid)")

        # Compile model
        print("\n[5.2] Compiling Model")
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        print("Optimizer: Adam")
        print("Loss Function: Binary Cross-Entropy")
        print("Metrics: Accuracy, AUC")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            'models/neural_network.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Train model
        print("\n[5.3] Training Neural Network")
        print("Epochs: 100 (with Early Stopping, patience=15)")
        print("Validation Split: 20%")
        print("Batch Size: 32")

        start_time = time.time()
        history = model.fit(
            self.X_train_scaled,
            self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        nn_time = time.time() - start_time

        print(f"\nTraining completed in {nn_time:.2f}s")
        print(f"Total epochs trained: {len(history.history['loss'])}")

        # Evaluate
        print("\n[5.4] Evaluating Neural Network")
        y_pred_proba = model.predict(self.X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        nn_metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

        print(f"Test Set Performance:")
        print(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
        print(f"  Precision: {nn_metrics['precision']:.4f}")
        print(f"  Recall: {nn_metrics['recall']:.4f}")
        print(f"  F1-Score: {nn_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {nn_metrics['roc_auc']:.4f}")

        # Store results
        self.models['Neural Network'] = model
        self.results['Neural Network'] = nn_metrics
        self.training_times['Neural Network'] = nn_time

        # Plot training history
        self._plot_training_history(history)

        print("\n[STEP 5] COMPLETED: Neural Network trained successfully")

    def _plot_training_history(self, history):
        # Purpose: Visualize NN training curves (accuracy and loss) for diagnosis.
        # What: Reads the Keras History object and saves interactive plots to disk.
        # Why: Helps to inspect overfitting/underfitting and adjust training settings.
        # Params: history (keras.callbacks.History) returned by model.fit()
        print("\nGenerating training history plots...")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Model Loss')
        )

        # Accuracy plot
        fig.add_trace(
            go.Scatter(y=history.history['accuracy'], name='Train Accuracy',
                      mode='lines', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy',
                      mode='lines', line=dict(color='red')),
            row=1, col=1
        )

        # Loss plot
        fig.add_trace(
            go.Scatter(y=history.history['loss'], name='Train Loss',
                      mode='lines', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_loss'], name='Val Loss',
                      mode='lines', line=dict(color='red')),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        fig.update_layout(height=400, width=1000, title_text="Neural Network Training History")
        fig.write_html('results/nn_training_history.html')

        print("Training history saved to results/nn_training_history.html")

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        # Purpose: Compute standard classification and performance metrics.
        # What: Returns accuracy, precision, recall, f1, roc_auc, confusion matrix
        #       and classification report to summarize model results.
        # Why: These metrics provide both algorithmic and business-relevant
        #      performance measures to compare models.
        # Params:
        #   - y_true: ground truth labels
        #   - y_pred: predicted class labels
    #   - y_pred_proba: predicted probabilities for positive class
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }

    def perform_clustering(self):
        # Purpose: Segment customers into homogeneous groups using K-Means.
        # What: Selects business-relevant features, scales them, uses elbow method
        #       to select K, fits KMeans, and generates cluster profiles and visualizations.
        # Why: Segmentation supports targeted retention strategies and identifies
        #      high-value but at-risk groups for prioritization.
        # Params: Uses original dataframe self.df and clustering feature list:
        #   ['Age','Balance','Tenure','NumOfProducts','CreditScore']
        print("\n[STEP 6] CUSTOMER SEGMENTATION (K-MEANS CLUSTERING)")
        print("-" * 80)
        print("CO4: Unsupervised Learning - K-Means Clustering")

        # Select features for clustering
        clustering_features = ['Age', 'Balance', 'Tenure', 'NumOfProducts', 'CreditScore']
        print(f"\nClustering features: {clustering_features}")

        # Get original data (before SMOTE)
        X_original = self.df[clustering_features].copy()
        y_original = self.df['Exited'].copy()

        # Scale features
        scaler_cluster = StandardScaler()
        X_scaled = scaler_cluster.fit_transform(X_original)

        # Elbow method to find optimal K
        print("\n[6.1] Elbow Method - Finding Optimal K")
        inertias = []
        K_range = range(2, 8)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}")

        # Plot elbow curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(K_range),
            y=inertias,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=2)
        ))
        fig.update_layout(
            title='Elbow Method - Optimal K Selection',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia (Within-cluster sum of squares)',
            height=500,
            width=800
        )
        fig.write_html('results/cluster_plots/elbow_curve.html')
        print("Elbow curve saved to results/cluster_plots/elbow_curve.html")

        # Perform K-Means with optimal K (4 clusters)
        optimal_k = 4
        print(f"\n[6.2] Performing K-Means with K={optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to original data
        self.df['Cluster'] = cluster_labels

        # Create cluster profiles
        print("\n[6.3] Cluster Profiles")
        cluster_profiles = []

        for i in range(optimal_k):
            cluster_data = self.df[self.df['Cluster'] == i]
            profile = {
                'Cluster': i,
                'Size': len(cluster_data),
                'Avg_Age': cluster_data['Age'].mean(),
                'Avg_CreditScore': cluster_data['CreditScore'].mean(),
                'Avg_Balance': cluster_data['Balance'].mean(),
                'Avg_Tenure': cluster_data['Tenure'].mean(),
                'Avg_NumOfProducts': cluster_data['NumOfProducts'].mean(),
                'Avg_IsActiveMember': cluster_data['IsActiveMember'].mean(),
                'Churn_Rate': cluster_data['Exited'].mean() * 100
            }
            cluster_profiles.append(profile)

            # Get cluster interpretation
            name, description, strategies = interpret_cluster(pd.Series(profile), i)

            print(f"\nCluster {i}: {name}")
            print(f"  Size: {profile['Size']} customers ({profile['Size']/len(self.df)*100:.1f}%)")
            print(f"  Description: {description}")
            print(f"  Avg Age: {profile['Avg_Age']:.1f}")
            print(f"  Avg Balance: ${profile['Avg_Balance']:,.2f}")
            print(f"  Avg Tenure: {profile['Avg_Tenure']:.1f} years")
            print(f"  Avg Products: {profile['Avg_NumOfProducts']:.2f}")
            print(f"  Churn Rate: {profile['Churn_Rate']:.2f}%")

        # Save cluster profiles
        cluster_df = pd.DataFrame(cluster_profiles)
        cluster_df.to_csv('results/cluster_profiles.csv', index=False)
        print("\nCluster profiles saved to results/cluster_profiles.csv")

        # Create 3D visualization
        print("\n[6.4] Creating 3D Cluster Visualization")
        fig = px.scatter_3d(
            self.df,
            x='Age',
            y='Balance',
            z='Tenure',
            color='Cluster',
            size='Balance',
            hover_data=['CreditScore', 'NumOfProducts', 'Exited'],
            title='Customer Segmentation - 3D Cluster Visualization',
            labels={'Cluster': 'Cluster ID'},
            color_continuous_scale='Viridis'
        )
        fig.write_html('results/cluster_plots/3d_clusters.html')
        print("3D cluster visualization saved to results/cluster_plots/3d_clusters.html")

        # Cluster distribution
        fig = px.pie(
            cluster_df,
            values='Size',
            names='Cluster',
            title='Cluster Distribution'
        )
        fig.write_html('results/cluster_plots/cluster_distribution.html')

        print("\n[STEP 6] COMPLETED: Customer segmentation finished")

    def perform_association_rules(self):
        # Purpose: Discover frequent patterns and rules associated with churn.
        # What: Discretizes continuous features, converts rows to transaction lists,
        #       runs Apriori to find frequent itemsets and derives rules filtered
        #       for those predicting Exited=1 (churn).
        # Why: Association rules provide interpretable business insights to
        #      design targeted interventions (e.g., rule-based retention offers).
        # Params: min_support=0.02 and min_confidence=0.70 are used as thresholds.
        print("\n[STEP 7] ASSOCIATION RULE MINING")
        print("-" * 80)
        print("CO4: Unsupervised Learning - Pattern Discovery with Apriori Algorithm")

        # Discretize features
        print("\n[7.1] Discretizing Features for Association Rules")
        df_discrete = discretize_features(self.df)
        print("Features discretized into categorical bins")

        # Select relevant columns for association rules
        relevant_cols = [
            'CreditScore_Category', 'Age_Group', 'Balance_Category',
            'Tenure_Category', 'Salary_Category', 'Geography', 'Gender',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited'
        ]

        df_rules = df_discrete[relevant_cols].copy()

        # Convert to transactions format
        print("\n[7.2] Converting to Transaction Format")
        transactions = []
        for idx, row in df_rules.iterrows():
            transaction = []
            for col in relevant_cols:
                transaction.append(f"{col}={row[col]}")
            transactions.append(transaction)

        print(f"Created {len(transactions)} transactions")
        print(f"Sample transaction: {transactions[0]}")

        # Apply Apriori algorithm
        print("\n[7.3] Applying Apriori Algorithm")
        print("Parameters: min_support=0.02, min_confidence=0.70")

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Find frequent itemsets
        frequent_itemsets = apriori(
            df_encoded,
            min_support=0.02,
            use_colnames=True
        )

        print(f"Found {len(frequent_itemsets)} frequent itemsets")

        # Generate association rules
        if len(frequent_itemsets) > 0:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=0.70
            )

            # Filter for churn rules (Exited=1)
            churn_rules = rules[
                rules['consequents'].apply(lambda x: 'Exited=1' in str(x))
            ].copy()

            print(f"Generated {len(rules)} total rules")
            print(f"Found {len(churn_rules)} rules predicting churn (Exited=1)")

            # Sort by confidence
            churn_rules = churn_rules.sort_values('confidence', ascending=False)

            # Display top 15 rules
            print("\n[7.4] Top 15 Association Rules Predicting Churn")
            print("-" * 120)

            top_rules = churn_rules.head(15)
            for idx, rule in top_rules.iterrows():
                antecedent = str(rule['antecedents']).replace('frozenset({', '').replace('})', '').replace("'", "")
                consequent = str(rule['consequents']).replace('frozenset({', '').replace('})', '').replace("'", "")

                print(f"\nRule {idx+1}:")
                print(f"  IF: {antecedent}")
                print(f"  THEN: {consequent}")
                print(f"  Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.2f}")

                # Business interpretation
                if 'Geography=Germany' in antecedent and 'NumOfProducts=1' in antecedent:
                    print(f"  Insight: German customers with single product are high-risk")
                elif 'IsActiveMember=0' in antecedent:
                    print(f"  Insight: Inactive members have elevated churn risk")
                elif 'Age_Group=Senior' in antecedent:
                    print(f"  Insight: Senior customers require special attention")

            # Save rules to CSV
            churn_rules.to_csv('results/association_rules.csv', index=False)
            print("\n\nAssociation rules saved to results/association_rules.csv")

            # Create visualization
            fig = px.bar(
                top_rules.head(10),
                x='confidence',
                y=[str(ant)[:50] for ant in top_rules.head(10)['antecedents']],
                title='Top 10 Association Rules by Confidence',
                labels={'x': 'Confidence', 'y': 'Rule Antecedent'},
                orientation='h'
            )
            fig.write_html('results/association_rules_chart.html')
            print("Association rules chart saved to results/association_rules_chart.html")

        else:
            print("No frequent itemsets found. Try lowering min_support.")

        print("\n[STEP 7] COMPLETED: Association rule mining finished")

    def evaluate_and_compare_models(self):
        # Purpose: Aggregate model metrics, compare models and produce evaluation plots.
        # What: Builds a comparison table, identifies best models by recall and AUC,
        #       and triggers visualizations (ROC, confusion matrices, business cost).
        # Why: Enables selection of the operational model(s) balancing business cost
        #      and detection performance (recall/ROC-AUC).
        # Params: Operates on self.results and self.models populated by training steps.
        print("\n[STEP 8] MODEL EVALUATION AND COMPARISON")
        print("-" * 80)

        # Create comparison table
        print("\n[8.1] Model Performance Comparison")
        comparison_data = []

        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Training Time (s)': f"{self.training_times[name]:.2f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Save comparison
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print("\nModel comparison saved to results/model_comparison.csv")

        # Find best model
        best_recall_model = max(self.results.items(), key=lambda x: x[1]['recall'])
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])

        print(f"\n[8.2] Best Models")
        print(f"Best Recall (Most Important): {best_recall_model[0]} - {best_recall_model[1]['recall']:.4f}")
        print(f"Best ROC-AUC: {best_auc_model[0]} - {best_auc_model[1]['roc_auc']:.4f}")

        # Generate evaluation plots
        self._generate_evaluation_plots()

        print("\n[STEP 8] COMPLETED: Model evaluation finished")

    def _generate_evaluation_plots(self):
        # Purpose: Create a suite of visual artifacts for model comparison and business reporting.
        # What: Produces metrics comparison bars, ROC curves, confusion matrices and
        #       business cost charts and writes them to the results/ directory.
        # Why: Visual outputs support stakeholders and facilitate model selection.
        # Params: Uses self.results and self.models populated previously.
        print("\n[8.3] Generating Evaluation Plots")

        # 1. Metrics comparison bar chart
        metrics_data = []
        for name, result in self.results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'ROC-AUC': result['roc_auc']
            })

        df_metrics = pd.DataFrame(metrics_data)

        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_metrics['Model'],
                y=df_metrics[metric]
            ))

        fig.update_layout(
            title='Model Performance Comparison - All Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=600,
            width=1200
        )
        fig.write_html('results/metrics_comparison.html')

        # 2. ROC Curves
        fig = go.Figure()

        for name, model in self.models.items():
            if name == 'Neural Network':
                y_pred_proba = model.predict(self.X_test_scaled).flatten()
            else:
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{name} (AUC={auc_score:.3f})',
                mode='lines'
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='ROC Curves - All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=700,
            width=900
        )
        fig.write_html('results/roc_curves/all_models_roc.html')

        # 3. Confusion matrices
        for name, result in self.results.items():
            cm = result['confusion_matrix']

            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                text_auto=True,
                title=f'Confusion Matrix - {name}'
            )
            fig.write_html(f'results/confusion_matrices/{name.replace(" ", "_").lower()}_cm.html')

        # 4. Business cost analysis
        cost_data = []
        for name, result in self.results.items():
            cm = result['confusion_matrix']
            costs = calculate_business_cost(cm)
            cost_data.append({
                'Model': name,
                'Total Cost': costs['total_cost'],
                'FN Cost': costs['false_negative_cost'],
                'FP Cost': costs['false_positive_cost'],
                'Savings': costs['true_positive_savings'],
                'Net Impact': costs['net_impact']
            })

        df_costs = pd.DataFrame(cost_data)

        fig = px.bar(
            df_costs,
            x='Model',
            y='Total Cost',
            title='Business Cost Comparison (Lower is Better)',
            labels={'Total Cost': 'Total Business Cost ($)'},
            color='Total Cost',
            color_continuous_scale='Reds_r'
        )
        fig.write_html('results/business_cost_comparison.html')

        print("All evaluation plots generated and saved to results/")

    def save_models(self):
        # Purpose: Persist trained models and preprocessing artifacts for deployment.
        # What: Saves scikit-learn models (joblib), the scaler, label encoder and
        #       a JSON summary of results. Neural network model is saved during training.
        # Why: Allows the Streamlit app and other systems to load models without
        #      retraining, enabling fast inference in production.
        # Params: None. Reads from self.models, self.scaler and self.feature_names.
        print("\n[STEP 9] SAVING MODELS AND ARTIFACTS")
        print("-" * 80)

        # Save traditional models
        for name, model in self.models.items():
            if name != 'Neural Network':  # Neural network already saved during training
                filename = f"models/{name.replace(' ', '_').lower()}.pkl"
                joblib.dump(model, filename)
                print(f"Saved: {filename}")

        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Saved: models/scaler.pkl")

        # Save label encoder
        joblib.dump(self.label_encoder_gender, 'models/label_encoder_gender.pkl')
        print("Saved: models/label_encoder_gender.pkl")

        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        print("Saved: models/feature_names.pkl")

        # Save results summary
        results_summary = {
            'models': {},
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(self.df),
                'churn_rate': self.df['Exited'].mean() * 100,
                'features': len(self.feature_names)
            }
        }

        for name, metrics in self.results.items():
            results_summary['models'][name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1']),
                'roc_auc': float(metrics['roc_auc']),
                'training_time': float(self.training_times[name])
            }

        with open('results/results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        print("Saved: results/results_summary.json")

        print("\n[STEP 9] COMPLETED: All models and artifacts saved successfully")

    def run_complete_pipeline(self):
        # Purpose: Orchestrate end-to-end execution of the training pipeline.
        # What: Sequentially runs data loading, preprocessing, model training,
        #       hyperparameter tuning, neural network training, clustering,
        #       association rules, evaluation and model saving.
        # Why: Provides a single-call entry point for reproducing all experiments
        #      and generating results/artifacts for reporting.
        # Params: None. Uses internal attributes and methods to carry out each step.
        print("\n\nSTARTING COMPLETE TRAINING PIPELINE")
        print("=" * 80)

        try:
            # CO2: Data preprocessing
            self.load_and_explore_data()
            self.preprocess_data()

            # CO3: Supervised learning
            self.train_traditional_models()

            # CO1: Hyperparameter optimization
            self.hyperparameter_tuning()

            # CO5: Neural networks
            self.train_neural_network()

            # CO4: Clustering
            self.perform_clustering()

            # CO4: Association rules
            self.perform_association_rules()

            # Evaluation
            self.evaluate_and_compare_models()

            # Save everything
            self.save_models()

            print("\n\n" + "=" * 80)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nAll 5 Course Outcomes (COs) have been demonstrated:")
            print("  CO1: AI-based Heuristic Techniques (Grid Search CV)")
            print("  CO2: Data Preprocessing (Scaling, Encoding, SMOTE, Feature Engineering)")
            print("  CO3: Supervised Learning (6 Classification Models)")
            print("  CO4: Unsupervised Learning (K-Means Clustering + Association Rules)")
            print("  CO5: Neural Networks (Deep Learning with TensorFlow)")
            print("\nNext Step: Run the Streamlit app")
            print("  Command: streamlit run app.py")
            print("=" * 80)

        except Exception as e:
            print(f"\n\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Initialize trainer
    trainer = BankChurnModelTrainer(data_path='data/Churn_Modelling.csv')

    # Run complete pipeline
    trainer.run_complete_pipeline()
