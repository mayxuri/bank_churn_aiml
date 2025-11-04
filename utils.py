"""
Utility Functions for Bank Customer Churn Prediction System
Contains helper functions used across training and prediction modules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def get_risk_category(probability: float) -> Tuple[str, str]:
    """
    Categorize churn probability into risk levels with associated colors.

    Args:
        probability: Churn probability between 0 and 1

    Returns:
        Tuple of (risk_category, color_class)
    """
    if probability < 0.30:
        return "LOW RISK", "risk-low"
    elif probability < 0.60:
        return "MEDIUM RISK", "risk-medium"
    else:
        return "HIGH RISK", "risk-high"


def format_currency(amount: float) -> str:
    """
    Format numerical value as currency.

    Args:
        amount: Numerical amount

    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def calculate_business_cost(cm: np.ndarray, fn_cost: float = 1500, fp_cost: float = 50) -> Dict[str, float]:
    """
    Calculate business costs from confusion matrix.

    In churn prediction:
    - False Negative (FN): Predicted no churn, but customer left - HIGHEST COST (lost revenue)
    - False Positive (FP): Predicted churn, but customer stayed - LOW COST (wasted retention offer)
    - True Negative (TN): Correctly predicted no churn - NO COST
    - True Positive (TP): Correctly predicted churn - SAVINGS (retention success)

    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]]
        fn_cost: Cost of missing a churner (lost lifetime value) - Default $1,500
        fp_cost: Cost of false alarm (wasted retention offer) - Default $50

    Returns:
        Dictionary with cost breakdown
    """
    tn, fp, fn, tp = cm.ravel()

    fn_total_cost = fn * fn_cost  # Missed churners - lost revenue
    fp_total_cost = fp * fp_cost  # Wasted retention offers
    total_cost = fn_total_cost + fp_total_cost

    # Calculate potential savings from correct predictions
    tp_savings = tp * fn_cost  # Successfully identified and retained churners

    return {
        'false_negative_cost': fn_total_cost,
        'false_positive_cost': fp_total_cost,
        'total_cost': total_cost,
        'true_positive_savings': tp_savings,
        'net_impact': tp_savings - total_cost,
        'false_negatives': int(fn),
        'false_positives': int(fp),
        'true_positives': int(tp),
        'true_negatives': int(tn)
    }


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretize continuous features into categorical bins for association rule mining.

    Business-driven binning strategy:
    - Credit Score: Poor (<600), Fair (600-700), Good (>700)
    - Age: Young (<35), Middle (35-50), Senior (>50)
    - Balance: Low (<50k), Medium (50k-100k), High (>100k)
    - Tenure: New (0-3 years), Established (4-6 years), Long-term (7-10 years)
    - EstimatedSalary: Low (<75k), Medium (75k-125k), High (>125k)

    Args:
        df: DataFrame with continuous features

    Returns:
        DataFrame with discretized features
    """
    df_discrete = df.copy()

    # Credit Score categories
    df_discrete['CreditScore_Category'] = pd.cut(
        df['CreditScore'],
        bins=[0, 600, 700, 1000],
        labels=['Poor', 'Fair', 'Good']
    )

    # Age groups
    df_discrete['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 35, 50, 100],
        labels=['Young', 'Middle', 'Senior']
    )

    # Balance categories
    df_discrete['Balance_Category'] = pd.cut(
        df['Balance'],
        bins=[-1, 50000, 100000, 300000],
        labels=['Low', 'Medium', 'High']
    )

    # Tenure categories
    df_discrete['Tenure_Category'] = pd.cut(
        df['Tenure'],
        bins=[-1, 3, 6, 11],
        labels=['New', 'Established', 'LongTerm']
    )

    # Salary categories
    df_discrete['Salary_Category'] = pd.cut(
        df['EstimatedSalary'],
        bins=[0, 75000, 125000, 300000],
        labels=['Low', 'Medium', 'High']
    )

    # Convert to string for association rules
    categorical_columns = [
        'CreditScore_Category', 'Age_Group', 'Balance_Category',
        'Tenure_Category', 'Salary_Category', 'Geography', 'Gender',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited'
    ]

    for col in categorical_columns:
        if col in df_discrete.columns:
            df_discrete[col] = df_discrete[col].astype(str)

    return df_discrete


def interpret_cluster(cluster_stats: pd.Series, cluster_id: int) -> Tuple[str, str, List[str]]:
    """
    Interpret cluster characteristics and assign business-meaningful names and strategies.

    Args:
        cluster_stats: Series containing average values for cluster
        cluster_id: Cluster identifier

    Returns:
        Tuple of (cluster_name, description, retention_strategies)
    """
    avg_balance = cluster_stats.get('Balance', 0)
    avg_products = cluster_stats.get('NumOfProducts', 0)
    avg_active = cluster_stats.get('IsActiveMember', 0)
    avg_tenure = cluster_stats.get('Tenure', 0)
    churn_rate = cluster_stats.get('Churn_Rate', 0)

    # Define cluster profiles based on characteristics
    if avg_balance > 100000 and avg_products >= 2 and avg_active > 0.7:
        name = "Premium Loyalists"
        description = f"High-value customers (avg balance: ${avg_balance:,.0f}) with multiple products and high engagement. Low churn risk ({churn_rate:.1f}%)."
        strategies = [
            "Maintain VIP status with exclusive benefits",
            "Offer premium financial advisory services",
            "Provide early access to new investment products",
            "Assign dedicated relationship managers"
        ]

    elif avg_balance > 80000 and avg_active < 0.4 and avg_products < 2:
        name = "At-Risk High-Value"
        description = f"High balance (${avg_balance:,.0f}) but low engagement and single product. Critical churn risk ({churn_rate:.1f}%)."
        strategies = [
            "URGENT: Launch personalized re-engagement campaign",
            "Offer product bundling with attractive discounts",
            "Schedule proactive financial review meetings",
            "Implement win-back offers before they leave"
        ]

    elif avg_tenure > 6 and churn_rate < 15:
        name = "Stable Long-Term"
        description = f"Long-standing customers (avg {avg_tenure:.1f} years tenure) with stable relationship. Reliable segment ({churn_rate:.1f}% churn)."
        strategies = [
            "Reward loyalty with tenure-based benefits",
            "Cross-sell additional products",
            "Maintain regular communication",
            "Anniversary recognition programs"
        ]

    elif avg_balance < 30000 and avg_active < 0.5:
        name = "Dormant Accounts"
        description = f"Low balance (${avg_balance:,.0f}) and inactive. Highest churn risk ({churn_rate:.1f}%)."
        strategies = [
            "Cost-effective digital re-activation campaigns",
            "Special promotions to increase balance",
            "Educational content on product benefits",
            "Consider account maintenance fee waivers"
        ]

    elif avg_products >= 2 and avg_active > 0.6:
        name = "Engaged Multi-Product"
        description = f"Active customers with {avg_products:.1f} products on average. Good retention ({churn_rate:.1f}% churn)."
        strategies = [
            "Upsell premium versions of existing products",
            "Introduce complementary services",
            "Loyalty rewards for multi-product usage",
            "Referral incentive programs"
        ]

    else:
        name = "Standard Customers"
        description = f"Average profile across metrics. Moderate churn risk ({churn_rate:.1f}%)."
        strategies = [
            "Standard retention programs",
            "Targeted product recommendations",
            "Regular satisfaction surveys",
            "Gradual engagement improvement initiatives"
        ]

    return name, description, strategies


def format_association_rule(antecedent: str, consequent: str, support: float,
                           confidence: float, lift: float) -> str:
    """
    Format association rule for display.

    Args:
        antecedent: Left side of rule (IF conditions)
        consequent: Right side of rule (THEN result)
        support: Rule support
        confidence: Rule confidence
        lift: Rule lift

    Returns:
        Formatted rule string
    """
    ant_clean = antecedent.replace('frozenset({', '').replace('})', '').replace("'", "")
    cons_clean = consequent.replace('frozenset({', '').replace('})', '').replace("'", "")

    return f"IF {ant_clean} THEN {cons_clean} (Conf: {confidence:.1%}, Supp: {support:.3f}, Lift: {lift:.2f})"


def get_retention_recommendations(features: Dict, churn_prob: float) -> List[str]:
    """
    Generate personalized retention recommendations based on customer features.

    Args:
        features: Dictionary of customer features
        churn_prob: Predicted churn probability

    Returns:
        List of actionable retention strategies
    """
    recommendations = []

    # Check number of products
    if features.get('NumOfProducts', 1) == 1:
        recommendations.append(
            "PRODUCT BUNDLING: Customer uses only 1 product. "
            "Offer 10-15% discount on additional products (savings account + credit card bundle)."
        )

    # Check activity status
    if features.get('IsActiveMember', 1) == 0:
        recommendations.append(
            "RE-ENGAGEMENT CAMPAIGN: Customer is inactive. "
            "Send personalized email/SMS highlighting unused features and exclusive reactivation offers."
        )

    # Check balance
    if features.get('Balance', 0) < 50000:
        recommendations.append(
            "FINANCIAL PLANNING: Low account balance detected. "
            "Provide free financial consultation to help grow savings and strengthen relationship."
        )

    # Check credit card
    if features.get('HasCrCard', 1) == 0:
        recommendations.append(
            "CREDIT CARD OFFER: No credit card detected. "
            "Promote cashback or rewards credit card with first-year fee waiver."
        )

    # Age-based recommendations
    age = features.get('Age', 40)
    if age > 50 and features.get('IsActiveMember', 1) == 0:
        recommendations.append(
            "SENIOR RELATIONSHIP MANAGEMENT: Inactive senior customer. "
            "Assign dedicated relationship manager for personalized service and retirement planning."
        )
    elif age < 35 and features.get('NumOfProducts', 1) == 1:
        recommendations.append(
            "MILLENNIAL ENGAGEMENT: Young customer with single product. "
            "Promote mobile banking features, digital wallets, and investment apps."
        )

    # Geography-based
    if features.get('Geography') == 'Germany':
        recommendations.append(
            "REGIONAL STRATEGY: German market shows higher churn. "
            "Implement Germany-specific retention program with localized benefits."
        )

    # Tenure-based
    if features.get('Tenure', 5) < 2:
        recommendations.append(
            "NEW CUSTOMER ONBOARDING: Low tenure indicates new customer. "
            "Enhance onboarding experience with welcome bonuses and educational content."
        )

    # High-risk urgent actions
    if churn_prob > 0.7:
        recommendations.insert(0,
            "URGENT INTERVENTION REQUIRED: Very high churn risk! "
            "Immediate outreach by retention specialist within 48 hours. "
            "Authorize special retention offers up to $200 value."
        )

    # If no specific recommendations, provide general advice
    if len(recommendations) == 0:
        recommendations.append(
            "STANDARD RETENTION: Maintain regular communication and monitor satisfaction levels."
        )

    return recommendations


def calculate_clv(balance: float, num_products: int, tenure: int,
                 base_clv: float = 1500) -> float:
    """
    Calculate estimated Customer Lifetime Value (CLV).

    Simple CLV formula incorporating customer characteristics:
    CLV = Base_CLV * (1 + balance_factor) * product_factor * tenure_factor

    Args:
        balance: Customer account balance
        num_products: Number of products
        tenure: Years with bank
        base_clv: Base lifetime value

    Returns:
        Estimated CLV
    """
    # Balance factor: Higher balance = higher value
    balance_factor = min(balance / 100000, 2.0)  # Cap at 2x multiplier

    # Product factor: More products = higher value
    product_factor = 1 + (num_products - 1) * 0.3  # +30% per additional product

    # Tenure factor: Longer tenure = higher value
    tenure_factor = 1 + (tenure / 10) * 0.5  # +5% per year, max 50%

    clv = base_clv * (1 + balance_factor) * product_factor * tenure_factor

    return clv


def create_cluster_name_mapping(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Create mapping from cluster IDs to business-meaningful names.

    Args:
        cluster_profiles: DataFrame with cluster statistics

    Returns:
        Dictionary mapping cluster_id to cluster_name
    """
    cluster_mapping = {}

    for idx, row in cluster_profiles.iterrows():
        name, _, _ = interpret_cluster(row, idx)
        cluster_mapping[idx] = name

    return cluster_mapping


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw customer data.

    Feature Engineering Strategy:
    1. Ratio features: Capture relative relationships
    2. Categorical features: Group continuous values into business-meaningful categories
    3. Interaction features: Combine multiple features

    Args:
        df: Raw customer dataframe

    Returns:
        DataFrame with additional engineered features
    """
    df_eng = df.copy()

    # Ratio features - capture relative financial health
    # Avoid division by zero
    df_eng['BalanceSalaryRatio'] = df_eng['Balance'] / (df_eng['EstimatedSalary'] + 1)
    df_eng['TenureAgeRatio'] = df_eng['Tenure'] / (df_eng['Age'] + 1)
    df_eng['BalancePerProduct'] = df_eng['Balance'] / (df_eng['NumOfProducts'] + 1)

    # Age categories
    df_eng['AgeGroup'] = pd.cut(
        df_eng['Age'],
        bins=[0, 35, 50, 100],
        labels=['Young', 'Middle', 'Senior']
    )

    # Balance categories
    df_eng['BalanceCategory'] = pd.cut(
        df_eng['Balance'],
        bins=[-1, 50000, 100000, 300000],
        labels=['Low', 'Medium', 'High']
    )

    # Credit Score categories
    df_eng['CreditScoreCategory'] = pd.cut(
        df_eng['CreditScore'],
        bins=[0, 600, 700, 1000],
        labels=['Poor', 'Fair', 'Good']
    )

    return df_eng


def validate_input_data(data: Dict) -> Tuple[bool, str]:
    """
    Validate user input data for prediction.

    Args:
        data: Dictionary of input features

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Define valid ranges
    validations = {
        'CreditScore': (350, 850),
        'Age': (18, 92),
        'Tenure': (0, 10),
        'Balance': (0, 250000),
        'NumOfProducts': (1, 4),
        'EstimatedSalary': (0, 200000)
    }

    for field, (min_val, max_val) in validations.items():
        if field in data:
            if not (min_val <= data[field] <= max_val):
                return False, f"{field} must be between {min_val} and {max_val}"

    # Categorical validations
    if 'Geography' in data and data['Geography'] not in ['France', 'Spain', 'Germany']:
        return False, "Geography must be France, Spain, or Germany"

    if 'Gender' in data and data['Gender'] not in ['Male', 'Female']:
        return False, "Gender must be Male or Female"

    return True, ""
