# Dataset Directory

## Required Dataset

Place the `Churn_Modelling.csv` file in this directory.

### Download Instructions

1. Go to Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
2. Download the dataset (you may need to create a free Kaggle account)
3. Extract `Churn_Modelling.csv` from the downloaded ZIP file
4. Place it in this directory

### Expected File

- **Filename**: `Churn_Modelling.csv`
- **Size**: Approximately 2.5 MB
- **Rows**: 10,000 customers
- **Columns**: 13 features + 1 target variable

### Dataset Columns

| Column Name | Description |
|-------------|-------------|
| CustomerId | Unique identifier for each customer |
| Surname | Customer's last name |
| CreditScore | Credit score (350-850) |
| Geography | Country (France, Spain, Germany) |
| Gender | Male or Female |
| Age | Customer age (18-92) |
| Tenure | Years with bank (0-10) |
| Balance | Account balance |
| NumOfProducts | Number of products (1-4) |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Active member (0/1) |
| EstimatedSalary | Estimated salary |
| **Exited** | **TARGET: Churned (0/1)** |

### Verification

After placing the file, verify it's correct:

```python
import pandas as pd

df = pd.read_csv('data/Churn_Modelling.csv')
print(df.shape)  # Should print: (10000, 14)
print(df.columns.tolist())
```

Expected output:
```
(10000, 14)
['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
```

### Note

The dataset is NOT included in this repository due to size and licensing.
You must download it separately from Kaggle.
