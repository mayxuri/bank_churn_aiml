# Bank Customer Churn Prediction - Research Paper Flowchart

## Methodology Flowchart (Academic Style)

### Main System Architecture

```mermaid
flowchart TD
    Start([Bank Customer Data<br/>10,000 Records]) --> DataCollection[Data Collection<br/>Churn_Modelling.csv<br/>13 Features]

    DataCollection --> Preprocessing[Data Preprocessing<br/>• Missing value handling<br/>• Feature encoding<br/>• Feature scaling]

    Preprocessing --> FeatEng[Feature Engineering<br/>• Ratio features<br/>• Categorical binning<br/>• Derived attributes]

    FeatEng --> Split[Train-Test Split<br/>80% Train / 20% Test<br/>Stratified sampling]

    Split --> SMOTE[Class Balancing<br/>SMOTE Technique]

    SMOTE --> EDA[Exploratory Data Analysis<br/>• Correlation analysis<br/>• Distribution plots<br/>• Churn patterns]

    EDA --> ModelBuilding{Model Building}

    ModelBuilding --> Traditional[Traditional ML Models]
    ModelBuilding --> DeepL[Deep Learning]
    ModelBuilding --> Unsupervised[Unsupervised Learning]

    Traditional --> Models[• Logistic Regression<br/>• Decision Tree<br/>• Random Forest<br/>• XGBoost<br/>• SVM<br/>• Gradient Boosting]

    DeepL --> NN[Neural Network<br/>4 Dense Layers<br/>BatchNorm + Dropout]

    Unsupervised --> Cluster[K-Means Clustering<br/>Customer Segmentation]
    Unsupervised --> Assoc[Association Rules<br/>Apriori Algorithm]

    Models --> Optimization[Hyperparameter Tuning<br/>Grid Search CV]
    NN --> Optimization

    Optimization --> Evaluation[Model Evaluation<br/>• Accuracy<br/>• Precision<br/>• Recall<br/>• F1-Score<br/>• ROC-AUC]

    Cluster --> Analysis[Business Analytics]
    Assoc --> Analysis

    Evaluation --> Compare[Model Comparison<br/>Performance Metrics]

    Compare --> BestModel[Best Model Selection<br/>Random Forest Optimized]

    BestModel --> Deployment[Deployment<br/>Web Application]
    Analysis --> Deployment

    Deployment --> Prediction{Churn Prediction}

    Prediction -->|High Risk| Churn[Customer Will Churn<br/>Retention Strategy]
    Prediction -->|Low Risk| NoChurn[Customer Retained<br/>Standard Care]

    Churn --> End([Business Decision])
    NoChurn --> End

    style Start fill:#e3f2fd
    style End fill:#c8e6c9
    style ModelBuilding fill:#fff3e0
    style Prediction fill:#ffe0b2
    style BestModel fill:#c8e6c9
```

---

## Simplified Linear Flow (For Paper)

```mermaid
flowchart LR
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[EDA]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Best Model]
    G --> H[Deployment]

    style A fill:#bbdefb
    style B fill:#c5cae9
    style C fill:#d1c4e9
    style D fill:#f8bbd0
    style E fill:#ffccbc
    style F fill:#c5e1a5
    style G fill:#a5d6a7
    style H fill:#80cbc4
```

---

## Three-Phase Methodology

```mermaid
flowchart TD
    subgraph Phase1 [Phase 1: Data Preparation]
        DC[Data Collection] --> DP[Data Preprocessing]
        DP --> FE[Feature Engineering]
        FE --> CB[Class Balancing]
    end

    subgraph Phase2 [Phase 2: Model Development]
        MT[Model Training<br/>7 ML Algorithms] --> HT[Hyperparameter<br/>Tuning]
        HT --> ME[Model Evaluation]
        CL[Clustering] --> AR[Association<br/>Rules]
    end

    subgraph Phase3 [Phase 3: Deployment]
        MS[Model Selection] --> WA[Web Application]
        WA --> CP[Churn Prediction]
        CP --> RS[Retention Strategy]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3

    style Phase1 fill:#e3f2fd
    style Phase2 fill:#fff3e0
    style Phase3 fill:#c8e6c9
```

---

## Model Comparison Architecture

```mermaid
flowchart TD
    Data[Preprocessed Data] --> Split{Train/Test Split}

    Split --> Train[Training Set<br/>80%]
    Split --> Test[Test Set<br/>20%]

    Train --> LR[Logistic Regression]
    Train --> DT[Decision Tree]
    Train --> RF[Random Forest]
    Train --> XGB[XGBoost]
    Train --> SVM[SVM]
    Train --> GB[Gradient Boosting]
    Train --> NN[Neural Network]

    LR --> Eval[Performance Evaluation]
    DT --> Eval
    RF --> Eval
    XGB --> Eval
    SVM --> Eval
    GB --> Eval
    NN --> Eval

    Test --> Eval

    Eval --> Metrics[Accuracy: 87.2%<br/>Recall: 50.9%<br/>ROC-AUC: 86.7%]

    Metrics --> Best[Random Forest<br/>Best Performer]

    style Data fill:#e3f2fd
    style Best fill:#c8e6c9
    style Metrics fill:#fff9c4
```

---

## Decision Flow (For Results Section)

```mermaid
flowchart TD
    Input[Customer Input Data] --> Model[Churn Prediction Model]

    Model --> Score{Churn<br/>Probability}

    Score -->|> 60%| High[HIGH RISK<br/>Immediate Action]
    Score -->|30-60%| Medium[MEDIUM RISK<br/>Proactive Campaign]
    Score -->|< 30%| Low[LOW RISK<br/>Standard Care]

    High --> Action1[• Call within 48h<br/>• Premium offers<br/>• Retention specialist]
    Medium --> Action2[• Email/SMS campaign<br/>• Product bundles<br/>• Loyalty rewards]
    Low --> Action3[• Regular communication<br/>• Satisfaction surveys<br/>• Upselling]

    Action1 --> ROI[ROI: 2,900%<br/>Cost Savings]
    Action2 --> ROI
    Action3 --> ROI

    style High fill:#ffcdd2
    style Medium fill:#fff9c4
    style Low fill:#c8e6c9
    style ROI fill:#a5d6a7
```

---

## Course Outcomes Mapping (For Academic Requirements)

```mermaid
flowchart LR
    System[Bank Churn<br/>Prediction System]

    System --> CO1[CO1: AI Heuristics<br/>Grid Search CV]
    System --> CO2[CO2: Preprocessing<br/>SMOTE, Scaling]
    System --> CO3[CO3: Supervised<br/>7 Classifiers]
    System --> CO4[CO4: Unsupervised<br/>Clustering + Rules]
    System --> CO5[CO5: Neural Networks<br/>Deep Learning]

    style CO1 fill:#ffebee
    style CO2 fill:#e8f5e9
    style CO3 fill:#e3f2fd
    style CO4 fill:#fff3e0
    style CO5 fill:#f3e5f5
```

---

## Key Metrics Table

| Component | Specification |
|-----------|--------------|
| Dataset | 10,000 customers, 13 features |
| Preprocessing | SMOTE, StandardScaler, OneHotEncoder |
| Models Trained | 9 (6 traditional + 1 optimized RF + 1 optimized XGB + 1 NN) |
| Best Model | Random Forest (Optimized) |
| Accuracy | 87.2% |
| Precision | 79.8% |
| Recall | 50.9% |
| F1-Score | 62.1% |
| ROC-AUC | 86.7% |
| Customer Segments | 4 clusters |
| Business ROI | 2,900% |

---

## Recommended Flowchart for Research Paper

**Use the "Three-Phase Methodology"** or **"Simplified Linear Flow"** for your research paper as they are:
- ✅ Clean and professional
- ✅ Easy to understand
- ✅ Suitable for academic publications
- ✅ Show clear methodology
- ✅ Not cluttered with excessive detail

For **detailed technical sections**, use the "Main System Architecture" flowchart.
