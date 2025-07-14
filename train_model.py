#!/usr/bin/env python
# coding: utf-8

# In[31]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import mlflow
import os


# In[32]:


# Load dataset
df = pd.read_csv("data/Grocery_Customer_Churn_Data.csv")


# In[33]:


# 1. Data Preprocessing
# Handle negative values
df['days_since_last_purchase'] = df['days_since_last_purchase'].abs()  # Fix negative days
df['is_negative_sales'] = df['total_sales'] < 0  # Flag negative sales
df['total_sales'] = df['total_sales'].abs()  # Convert to positive


# In[34]:


# Handle missing values
df['avg_purchase_value'].fillna(df['avg_purchase_value'].median(), inplace=True)
df['promotion_type'].fillna('None', inplace=True)
df['purchase_frequency'].fillna('Unknown', inplace=True)


# In[35]:


# Drop irrelevant columns
df = df.drop(['customer_id', 'transaction_id', 'transaction_date', 'last_purchase_date'], axis=1)


# In[36]:


# Encode categorical variables
categorical_cols = ['gender', 'income_bracket', 'marital_status', 'education_level', 
                    'occupation', 'product_category', 'purchase_frequency', 'promotion_type']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[37]:


# Scale numerical features
numerical_cols = ['age', 'membership_years', 'quantity', 'unit_price', 'avg_purchase_value', 
                  'total_sales', 'total_transactions', 'total_items_purchased', 
                  'avg_discount_used', 'online_purchases', 'in_store_purchases', 
                  'days_since_last_purchase']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])


# In[38]:


# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')


# In[39]:


# Save cleaned dataset
df_encoded.to_csv("data/cleaned_grocery_churn_data.csv", index=False)
print("Cleaned data saved as 'cleaned_grocery_churn_data.csv'")


# In[40]:


# 2. EDA
# Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')
plt.savefig('reports/churn_distribution.png')
plt.close()


# In[41]:


# Feature importance (preliminary correlation)
plt.figure(figsize=(15, 15))
sns.heatmap(df[numerical_cols + ['churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('reports/correlation_matrix.png')
plt.close()


# In[51]:


# 3. Model Training with XGBoost


X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']
X = pd.get_dummies(X, drop_first=True)


# In[52]:
# Save final feature names used by model
X.columns.to_series().to_csv("models/model_features.csv", index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X.dtypes)


# In[53]:


# Define and tune XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', enable_categorical=True)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[54]:


# Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best ROC-AUC score (CV):", grid_search.best_score_)


# In[55]:


# Evaluate
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))


# In[56]:


# Feature importance
importances = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10))


# In[57]:


# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances.head(10))
plt.title('Top 10 Feature Importances')
plt.savefig('reports/feature_importance.png')
plt.close()


# In[58]:


# Save model
joblib.dump(best_model, 'models/churn_model.pkl')
print("Model saved as 'churn_model.pkl'")


# In[ ]:




