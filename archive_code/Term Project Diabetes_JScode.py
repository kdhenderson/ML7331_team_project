# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:05:13 2025

@author: jaren
"""

import numpy as np
import pandas as pd



# =============================================================================
# # Reading in data
# df = pd.read_csv('C:/Users/jaren/OneDrive/Desktop/MSDS/Machine Learning 1/Term Project/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv')
# print(df.head())
# 
# # Display basic information and check for missing values
# print(df['readmitted'].info())  # Data types and non-null counts
# print(df.describe())
# print(df.isnull().sum())  # Check for missing values
# print(df.head())  # Preview the data
# 
# 
# ################################### CLEANING #####################################################
# # Replace '?' with NaN 
# df.replace('?', pd.NA, inplace=True)
# 
# # Check for columns with missing values
# missing_summary = df.isnull().sum()
# print(missing_summary[missing_summary > 0])  # Columns with missing data
# 
# 
# 
# # Drop columns that may not contribute to analysis
# df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)
# 
# 
# # Convert target variable to binary (if needed)
# df['readmitted_target'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
# 
# df['readmitted_target'].info()
# 
# 
# 
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Distribution of numerical columns
# df['time_in_hospital'].hist(bins=15)
# plt.title('Time in Hospital Distribution')
# plt.xlabel('Days')
# plt.ylabel('Frequency')
# plt.show()
# 
# 
# # Factorize all categorical variables
# categorical_columns = df.select_dtypes(include='object').columns
# for col in categorical_columns:
#     df[col], _ = pd.factorize(df[col])
# 
# print(df.info())  
# print(df.head())  
# 
# ###################################### EDA ################################################
# # Correlation heatmap
# corr = df.corr()
# 
# print(corr)
# # Save the correlation matrix to a CSV file
# corr.to_csv('C:/Users/jaren/OneDrive/Desktop/MSDS/Machine Learning 1/Term Project/correlation_matrix.csv', index=True)
# 
# 
# # Histogram for numeric variables
# df.hist(figsize=(15, 10), bins=20)
# plt.tight_layout()
# plt.show()
# 
# 
# 
# # Distribution of the target variable
# df['readmitted'].value_counts().plot(kind='bar')
# plt.title('Distribution of Readmitted')
# plt.xlabel('Readmitted (0=NO, 1=>30, 2=<=30)')
# plt.ylabel('Count')
# plt.show()
# 
# 
# # Correlation with the target variable
# correlation_with_target = df.corr()['readmitted'].sort_values(ascending=False)
# print(correlation_with_target)
# 
# #Features of interest: ['number_inpatient','number_diagnoses','number_emergency']
# 
# 
# import seaborn as sns
# # Pairplot to explore relationships
# sns.pairplot(df, vars=['number_inpatient','number_diagnoses','number_emergency'], hue='readmitted')
# plt.show()
# 
# sns.pairplot(df, vars=['num_lab_procedures','diag_1','diag_2','diag_3'], hue='readmitted')
# plt.show()
# 
# 
# 
# # Boxplot for categorical vs numeric
# sns.boxplot(x='readmitted', y='time_in_hospital', data=df)
# plt.title('Time in Hospital by Readmitted')
# plt.show()
# 
# 
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# # Quick feature importance analysis
# X = df.drop(['readmitted','readmitted_target'], axis=1)
# y = df['readmitted']
# 
# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)
# 
# # Feature importance
# feature_importance = pd.Series(model.feature_importances_, index=X.columns)
# feature_importance.sort_values(ascending=False, inplace=True)
# print(feature_importance)
# 
# # Drop low-importance features
# df = df.drop(columns=['payer_code', 'examide', 'citoglipton'])
# 
# 
# 
# # Analyzing top features
# sns.boxplot(x='readmitted', y='num_medications', data=df)
# plt.title('Lab Procedures by Readmission Status')
# plt.show()
# 
# print(df[['num_lab_procedures', 'num_medications', 'readmitted']].corr())
# print(df[['number_inpatient', 'number_outpatient', 'readmitted']].corr())
# 
# df['num_proc_inpatient'] = df['num_lab_procedures'] * df['number_inpatient']
# print(df[['num_proc_inpatient', 'readmitted']].corr())
# =============================================================================


####################### New analysis with one-hot encoding ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading in data
df = pd.read_csv('C:/Users/jaren/OneDrive/Desktop/MSDS/Machine Learning 1/Term Project/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv')
print(df.head())

# Display basic information and check for missing values
print(df['readmitted'].info())  # Data types and non-null counts
print(df.describe())
print(df.isnull().sum())  # Check for missing values
print(df.head())  # Preview the data


################################### CLEANING #####################################################
# Replace '?' with NaN 
df.replace('?', pd.NA, inplace=True)

# Check for columns with missing values
missing_summary = df.isnull().sum()
print(missing_summary[missing_summary > 0])  # Columns with missing data



# Drop columns that may not contribute to analysis
df.drop(['encounter_id', 'patient_nbr','weight','max_glu_serum','A1Cresult'], axis=1, inplace=True)


high_cardinality_columns = [col for col in df.columns if df[col].nunique() > 9]

# Step 1: Create frequency maps for all columns
frequency_maps = {col: df[col].value_counts() for col in high_cardinality_columns}

# Step 2: Apply frequency encoding
for col in high_cardinality_columns:
    df[f'{col}_encoded'] = df[col].map(frequency_maps[col])

# Apply encoding with fallback for unseen categories
for col in high_cardinality_columns:
    df[f'{col}_encoded'] = df[col].map(frequency_maps[col]).fillna(0)

print(df)

# Dropping old high cardinality columns
df.drop(high_cardinality_columns, axis=1, inplace=True)

# Identify all new columns that have 'encoded' in their names
high_cardinality_columns = [col for col in df.columns if 'encoded' in col or 'readmitted' in col]


categorical_columns = df.select_dtypes(include='object').columns
categorical_columns = categorical_columns[0:26]
print(categorical_columns)


#one-hot encoding to all categorical columns
df_encoded = pd.get_dummies(df.drop(columns=high_cardinality_columns + ['readmitted']), 
                            columns=categorical_columns, 
                            drop_first=False)

df_encoded = pd.concat([df[high_cardinality_columns] , df[['readmitted']], df_encoded], axis=1)


# updated dataset
print(df_encoded.info()) 
print(df_encoded.head()) 

## Remove duplicate columns
df_encoded = df_encoded.loc[:, ~df_encoded.T.duplicated()]

# Check for columns with missing values
missing_summary = df_encoded.isnull().sum()
print(missing_summary[missing_summary > 0])  # Columns with missing data


# Factorize all categorical variables

# Select all boolean columns
categorical_columns = list(df_encoded.select_dtypes(include='bool').columns)

# Append 'readmitted' column to the list (if it exists in the DataFrame)
if 'readmitted' in df_encoded.columns:
    categorical_columns.append('readmitted')

# Factorize    
for col in categorical_columns:
    df_encoded[col], mapping = pd.factorize(df_encoded[col])

print(df_encoded.info())  
print(df_encoded.head())  


columns_and_dtypes_df = df_encoded.dtypes.reset_index()
columns_and_dtypes_df.columns = ['Column', 'Data Type']
print(columns_and_dtypes_df)

Left_out = df[['readmitted','diabetesMed']]

# Factorize    
for col in Left_out:
    df_encoded[col], mapping = pd.factorize(df_encoded[col])


from sklearn.ensemble import RandomForestClassifier

# Quick feature importance analysis
X = df_encoded.drop(['readmitted'], axis=1)
y = df_encoded['readmitted']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)

for feature,importance in feature_importance.items():
     if importance >= .02:
         print(f"{feature}: {importance}")
         
# Save feature names with importance >= 0.02 into a list
important_features = feature_importance[feature_importance >= 0.02].index.tolist()

# Now, `important_features` contains the feature names
print(important_features)


sns.pairplot(df_encoded, vars= important_features, hue='readmitted')
plt.show()

df_reduced = df[important_features]

############################# PCA for Feature Reduction ##############################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Preprocessing - Standardize the Data
# Assuming X is your feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_reduced)  # Standardize the features

# Step 2: Fit PCA
pca = PCA(n_components=13)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Step 3: Inspect Results
explained_variance_ratio = pca.explained_variance_ratio_  # Variance explained by each component
n_components = pca.n_components_  # Number of components chosen

print(f"Number of components selected: {n_components}")
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"Total explained variance: {sum(explained_variance_ratio):.2f}")

# Step 4: Create a DataFrame for PCA-transformed data
X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

# Save for further analysis
X_pca_df.to_csv("pca_transformed_data.csv", index=False)

# Visualize explained variance
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, marker='o', linestyle='--')
plt.title("Explained Variance by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.show()


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.1)
plt.title("t-SNE Visualization")
plt.colorbar(label="Readmitted")
plt.show()


#####################################

from sklearn.model_selection import train_test_split

# Plot PC1 vs PC2 by response class

pc1 = X_pca[:, 0]  # First principal component
pc2 = X_pca[:, 1]  # Second principal component

Read_Reduced = df_encoded['readmitted']

pca_df = pd.DataFrame({
    'PC1': pc1,
    'PC2': pc2,
    'Readmitted': Read_Reduced
})

# Perform stratified sampling (10%)
pca_sample, _ = train_test_split(
    pca_df,
    test_size=0.9,  # Keep 10% of the data
    stratify=pca_df['Readmitted'],  # Stratify by the 'Readmitted' column
    random_state=1234
)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot with the stratified sample
sns.scatterplot(
    data=pca_sample,
    x='PC1',
    y='PC2',
    hue='Readmitted',
    palette='muted',
    alpha=0.7,
    ax=ax1
)
ax1.set_title('Scatter Plot (10% Stratified Sample)')
ax1.set_xlabel('Principal Component 1 (PC1)')
ax1.set_ylabel('Principal Component 2 (PC2)')
ax1.legend(title='Readmitted')

# Get axis limits from scatterplot
x_limits = ax1.get_xlim()
y_limits = ax1.get_ylim()

# KDE plot with full data, matching axis limits
sns.kdeplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Readmitted',
    palette='muted',
    ax=ax2,
    legend=True
)
ax2.set_title('KDE Plot (Matched Axis Limits)')
ax2.set_xlabel('Principal Component 1 (PC1)')
ax2.set_ylabel('')

# Apply matching axis limits
ax2.set_xlim(x_limits)
ax2.set_ylim(y_limits)

# Add a figure-wide title
fig.suptitle('PC1 vs. PC2 by Readmitted', fontsize=14)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to prevent overlap with suptitle
# plt.savefig('plots/PC1vPC2_by_readmitted.png')
plt.show()