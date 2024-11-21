# Customer Churn Prediction and Credit Scoring System

## **Overview**

This project develops a machine learning solution to predict customer churn for a consumer credit card portfolio and assigns credit scores based on churn probabilities. The system helps business managers identify at-risk customers and prioritize customer retention strategies.

## **Key Features**
- **Churn Prediction**: Uses the **XGBoost** algorithm to predict the likelihood of a customer leaving the credit card services.
- **Credit Scoring System**: Maps churn probabilities to a credit scoring range (300–850), similar to standard credit scoring systems (e.g., FICO).
- **High Accuracy**: The model achieves an impressive **98.9% accuracy**, ensuring reliable predictions.
- **Class Imbalance Handling**: Tackles the 16.07% churn rate imbalance using XGBoost's `scale_pos_weight` parameter.
- **Interpretability**: Outputs churn probabilities, predicted class labels, and credit scores for each customer.

## **Workflow**

1. **Data Preprocessing**:
    - Handled missing values and encoded categorical variables using `OneHotEncoder`.
    - Scaled numerical features and performed feature engineering (e.g., average transaction value).

2. **Model Training**:
    - Trained an XGBoost model with the objective `binary:logistic` to predict churn probabilities.
    - Used `scale_pos_weight` to address class imbalance.
    - Achieved **98.9% accuracy** on the test set.

3. **Credit Scoring**:
    - Mapped churn probabilities to a custom credit scoring range (300–850).
    - Integrated the results with customer data for actionable insights.

4. **Evaluation**:
    - Assessed the model using classification metrics such as precision, recall, F1-score, and ROC-AUC.

## **Dataset**

The dataset used in this project includes information about customer demographics, account activity, and transaction behavior. Key columns include:
- `Customer_Age`: Age of the customer.
- `Gender`: Gender of the customer.
- `Credit_Limit`: Credit limit assigned to the customer.
- `Total_Trans_Amt`: Total transaction amount.
- `Total_Trans_Ct`: Total transaction count.
- `Attrition_Flag`: Target variable indicating churn (attrited) or retention (existing).

## **Code Structure**

### **1. Preprocessing**
- Categorical features like `Education_Level`, `Income_Category`, and `Card_Category` are one-hot encoded.
- Numerical features like `Credit_Limit` are scaled and engineered into derived features (e.g., average transaction value).

### **2. Model Training**
- **XGBoost Classifier**:
    - Objective: `binary:logistic`
    - Class Imbalance Handling: `scale_pos_weight`
    - Trained on preprocessed data to predict churn probabilities.
    - Achieved **98.9% accuracy** on the test set.

### **3. Credit Scoring**
- Churn probabilities are mapped to credit scores using the formula:
    \[
    \text{Credit Score} = \text{min\_score} + (\text{max\_score} - \text{min\_score}) \times (1 - \text{Churn Probability})
    \]
- Scores range from **300** (high risk of churn) to **850** (low risk of churn).

### **4. Evaluation**
- Metrics used:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/churn-credit-scoring.git
   cd churn-credit-scoring
