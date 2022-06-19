#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%
df_merged = pd.read_excel("Series_dataset_2.xlsx")
df_merged.head()
#%%
df_merged.drop("ESGScore", axis=1, inplace = True)
#%%
df_merged.drop("Unnamed: 0", axis=1, inplace = True)
#%%
df_merged.head()
# %%
rule = lambda x: 1 if x>=1 else 0
controversy_cols = df_merged.iloc[:, 3:26]
controversy_cols
#%%
df_merged["Sum"] = controversy_cols.sum(axis=1)
df_merged["Label"] = controversy_cols.sum(axis=1).apply(rule)
#%%
# The imbalance between classes can be seen from here 
df_merged["Label"].value_counts()
df_merged.drop(controversy_cols.columns.values.tolist(), axis=1,inplace=True)
df_merged.head()
#%%
# High percentages mean more missing values in proportion
print(df_merged.isna().mean())  
# Identify the columns that have more than half of their values missing
df_merged[df_merged.columns[df_merged.isnull().mean() > 0.5]]
#%%
# Separate the features
feature_set = df_merged.iloc[:,2:56]
#feature_set["Health&SafetyPolicy"].iloc[0:10].replace(to_replace = 500, value = np.nan , inplace = True)
feature_set.head()

#%%
df_merged = df_merged[df_merged["Company Name"].notna()]
df_merged["Company Name"].isna().value_counts()
#%%
years = []
index_length = len(df_merged["Company Name"])//10
for i in range(index_length):
    for j in range(10):
        years.append(j)
df_merged.insert(1, "Fiscal Years", years)
df_merged.head()
#%%
#feature_set = feature_set.reset_index()
df_merged.set_index(["Company Name", "Fiscal Years"], inplace=True)
#%%
df_merged.head()

#%%
# Shape before dropping columns
keep_track = df_merged.shape
df_merged.loc["Zignago Vetro SpA",9].isnull().sum() 


#%%
controversy_companies = dict()
non_controversy_companies = dict()
counter = 0
nan_threshold = 90

df_merged_sorted = df_merged.copy()
df_merged_sorted.sort_index(level = "Company Name", inplace = True)

for idx,row in df_merged_sorted.iterrows():

    df_merged_grouped = df_merged_sorted.groupby(level = ["Company Name"])
    temp_df = df_merged_grouped.get_group(idx[0])

    # Count the number of controversies per company across all years
    controversy_sum = temp_df["Label"].sum() 
    percent_missing_value = temp_df.loc[idx].isnull().sum() * 100/(len(row)- 4)
    #print(f"Missing value percentage in FY{idx[1]}", " : " , percent_missing_value,"% ")
    
    if percent_missing_value > nan_threshold:
        df_merged_sorted.drop(index = idx, axis=0, inplace=True)
        
    keep_track = df_merged.shape
    counter = counter + 1

    print("Iteration number: ", counter)

#%%
df_merged_sorted.to_excel("Nans_removed_dataset.xlsx")    
print(controversy_companies)
print(non_controversy_companies)

#%%
# METHODOLOGY
# 1) Loop through the row values of the dataset
# 2) While looping, select a subset of dataset of the dataset that has the same index name (company)
# 3) If the ratio of NaN's to reported values is above a threshold drop the entire observation (this should be comparative based on the max())
# 4) The next step is to use the backward fill method provided by Pandas, which will 

df_merged_sorted.shape
df_merged_sorted.head(10)

#%%
companies_list = df_merged_sorted.index.get_level_values('Company Name').unique().tolist()
print(companies_list)
#%%
df_merged_sorted = pd.read_excel("Nans_removed_dataset.xlsx", index_col =  [0, 1])
#%%
# Check the number of missing values before filling
df_merged_sorted.isna().sum()
columns_list = list(df_merged_sorted.columns)
df_list = []
#%%
for company in companies_list:

    subset_df = df_merged_sorted.loc[company].bfill(axis = "rows")
    df_list.append(subset_df)

vertical_concat = pd.concat(df_list, axis=0)
vertical_concat.isna().sum() 
#%%
vertical_concat.head(10)

# %%
from sklearn.impute import KNNImputer
#%%
vertical_concat_features = vertical_concat.iloc[:,2:56]
# N_neighbors is a preprocessing hyperparameter
impute_knn = KNNImputer(n_neighbors=5)
knn_imputed_df = impute_knn.fit_transform(vertical_concat_features)

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std

# Drop the row that has NaN values (last row)
X = np.array(knn_imputed_df)
y = np.array(vertical_concat["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.3, random_state=42)
#%%
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
predictions = tree_clf.predict(X_test)

rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(X_train,y_train)
rf_predictions = rnd_clf.predict(X_test)

print("Random Forest Accuracy: ", accuracy_score(y_test, rf_predictions))
print("Random Forest Precision: ", precision_score(y_test, rf_predictions))
print("Random Forest Recall: ", recall_score(y_test, rf_predictions))
confusion_matrix(y_test,rf_predictions)

print(accuracy_score(y_test, predictions))
print(precision_score(y_test, predictions))
print(recall_score(y_test, predictions))
confusion_matrix(y_test,predictions)


#%%
pipeline = Pipeline(steps=[('i', impute_knn), ('m', rnd_clf)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='precision', cv=cv, n_jobs=-1, error_score='raise')
print('Mean Precision: %.3f (%.3f)' % (mean(scores), std(scores)))




# %%
from collections import Counter
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 42) 
X_res, y_res = sm.fit_resample(X_train, y_train)

print('Original dataset shape %s'%Counter(y_train))
print('Resampled dataset shape %s'%Counter(y_res))


# Test the newly populated dataset 
rf_clf_resampled = RandomForestClassifier()
rf_clf_resampled.fit(X_res,y_res)
rf_predictions_after_resampled = rf_clf_resampled.predict(X_test)

print(confusion_matrix(y_test, rf_predictions_after_resampled))
print(accuracy_score(y_test, rf_predictions_after_resampled))
print(precision_score(y_test, rf_predictions_after_resampled))
print(recall_score(y_test, rf_predictions_after_resampled))
