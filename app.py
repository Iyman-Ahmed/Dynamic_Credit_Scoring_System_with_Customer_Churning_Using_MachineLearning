# -*- coding: utf-8 -*-
"""app.ipynb

**Import Lib and Data**

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/path/to/dataset.csv')

data.head()

"""# **Data PreProcessing**"""

data.info()

data = data.drop(columns=['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
data.describe()

data.isnull().sum()

data.duplicated().sum()

sns.boxplot(data = data['Customer_Age'])
count = (data['Customer_Age'] >65).sum()
print(count)

sns.boxplot(data = data['Income_Category'])

count = data['Attrition_Flag'].value_counts()

plt.bar(count.index, count.values)
plt.xlabel('Attrition Flag')
plt.ylabel('Count')
plt.title('Attrition Flag Distribution')
plt.show()
count2 = data['Gender'].value_counts()

plt.bar(count2.index, count2.values)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = LabelEncoder()
onehot = OneHotEncoder()

columns = ['Attrition_Flag','Income_Category','Card_Category','Gender']
for col in columns:
  data[col] = encoder.fit_transform(data[col])

data = pd.get_dummies(data,columns=['Education_Level','Marital_Status'], drop_first=True).astype(int)



data.head()

data['Avg_Trans_value'] = data['Total_Trans_Amt']/data['Total_Trans_Ct']
data['Total_Chng_Q4_Q1'] = (data['Total_Amt_Chng_Q4_Q1']+ data['Total_Ct_Chng_Q4_Q1'])/2
data.replace([float('inf'), float('-inf')], 0, inplace=True)
data.fillna(0, inplace=True)
data.head()

from sklearn.model_selection import train_test_split
X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

y

"""# **ML Model Training**"""

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

"""***XGBoost Model***"""

xgbmodel = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgbmodel.fit(X_train, y_train)
xgb_pred = xgbmodel.predict(X_test)
xgb_preds_proba = xgbmodel.predict_proba(X_test)
churn_probabilities = xgb_preds_proba[:, 1]

print('XGB Classification Report:')
print(classification_report(y_test, xgb_pred))
print('XGB AUC Score:', roc_auc_score(y_test, churn_probabilities))
print(xgb_preds_proba)

def probability_to_score(probability, min_score=300, max_score=850):
        return min_score + (max_score - min_score) * (1 - probability)
credit_scores = [probability_to_score(prob) for prob in churn_probabilities]

result_df = X_test.copy()
result_df['Churn_Probability'] = churn_probabilities
result_df['Credit_Score'] = credit_scores
result_df['Actual_Churn'] = y_test
result_df.head()

