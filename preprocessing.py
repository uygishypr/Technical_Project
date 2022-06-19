
#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#%%
df_1 = pd.read_excel("dataset_controversies.xlsx")
df_2 = pd.read_excel("with_features_2.xlsx")
df_3 = pd.read_excel("with_features_3.xlsx")

#%%
columns_df_3 = list(df_3.columns.values)
social_score = df_3["Average of\nSocial Pillar Score\nOver the last 10 FY"]

#%%
# if sum of no controversies > threshold, change label to 1 
rule = lambda x: 1 if x>=3 else 0
controversy_cols = df_1.iloc[:, 3:21]

df_1["sums"] = controversy_cols.sum(axis=1)
df_1["label"] = controversy_cols.sum(axis=1).apply(rule)

controversy_cols

#%%
df_2_grouped = df_2.groupby("GICS Sector Name")
# Group data into sectors
financials_df = df_2_grouped.get_group('Financials')
healthcare_df = df_2_grouped.get_group('Health Care')
industrials_df = df_2_grouped.get_group('Industrials')
consumer_discretionary_df = df_2_grouped.get_group('Consumer Discretionary')
materials_df = df_2_grouped.get_group('Materials')
energy_df = df_2_grouped.get_group('Energy')
consumer_staples_df = df_2_grouped.get_group("Consumer Staples")

#%%
# Check the number of NaN values per column, for each sector to spot any reporting pattern
# Number of companies in this sector is 355, only 332 reported 
print(financials_df.isna().sum().idxmax())
print(healthcare_df.isna().sum().idxmax())
print(industrials_df.isna().sum().idxmax())
print(consumer_discretionary_df.isna().sum().idxmax())
print(materials_df.isna().sum().idxmax())
print(energy_df.isna().sum().idxmax())
print(consumer_staples_df.isna().sum().idxmax())

# Remove the features with the most frequently non-reported values
#%%
# High percentages mean more missing values in proportion
print(df_2.isna().mean())  

# Identify the columns that have more than half of their values missing
df_2[df_2.columns[df_2.isnull().mean() > 0.5]]

#%%
# Aggregate the TRUE FALSE values found in groups of 9
columns = list(df_2.columns.values)
cols_list = []
for column in columns:
    if "FY-" in column:
        cols_list.append(column)
        print(column)

isolated = df_2[df_2.columns[df_2.columns.isin(cols_list)]]

filter = lambda x: 1 if x>5 else 0

for i in range(0,len(cols_list),10):
    isolated["".join(cols_list[i].strip()[:-4]) + "Average"]= isolated.iloc[:,i:i+10].sum(axis=1).apply(filter)
    print(isolated[cols_list[i]])

# %%
list(isolated["Policy Bribery and Corruption Average"])
combined_df = isolated.iloc[:,318:-1]
combined_df["label"] = df_1["label"]
combined_df.insert(0, "Identifier", df_1["Identifier"])
combined_df.insert(1, "Company Name", df_1["Company Name"])
combined_df.insert(2, "GICS Sector Name", df_1["GICS Sector Name"])

# %%
# This cell for dropping that last row, only execute once
combined_df = combined_df.drop(combined_df.index[-1])

################################################################################
################################################################################
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Drop the row that has NaN values (last row)
X = np.array(combined_df.iloc[:,3:34])
y = np.array(combined_df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.3, random_state=42)
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
predictions = tree_clf.predict(X_test)

print(accuracy_score(y_test, predictions))
print(precision_score(y_test, predictions))
print(recall_score(y_test, predictions))
confusion_matrix(y_test,predictions)

# Compare to a base estimator 
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import VotingClassifier

rnd_clf = RandomForestClassifier(n_estimators = 100)
log_clf = LogisticRegression()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators =[('lr',log_clf),('rf', rnd_clf),('svc',svm_clf)],
    voting = 'hard'
)

voting_clf.fit(X_train,y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#%%
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train,y_train)
rf_predictions = rnd_clf.predict(X_test)

print(accuracy_score(y_test, rf_predictions))
print(precision_score(y_test, rf_predictions))
print(recall_score(y_test, rf_predictions))
confusion_matrix(y_test,rf_predictions)

#%%
combined_df["label"].value_counts()

#%% 
from collections import Counter
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 42) 
X_res, y_res = sm.fit_resample(X_train, y_train)

print('Original dataset shape %s'%Counter(y_train))
print('Resampled dataset shape %s'%Counter(y_res))


# Test the newly populated dataset 
rf_clf_resampled = SVC()
rf_clf_resampled.fit(X_res,y_res)
rf_predictions_after_resampled = rf_clf_resampled.predict(X_test)

print(confusion_matrix(y_test, rf_predictions_after_resampled))
print(accuracy_score(y_test, rf_predictions_after_resampled))
print(precision_score(y_test, rf_predictions_after_resampled))
print(recall_score(y_test, rf_predictions_after_resampled))

#%%
X_res.shape
X_test.shape

#%%

#data1 = df_1["sums"]
data2 = social_score


# Separate companies into 2 arrays, one companies with controversy
# other without controversy in the past 10 years.   

plt.figure(figsize=(8,6))

#plt.hist(data1, bins=100, alpha=0.5, label="data1")
plt.hist(data2, bins=100, alpha=0.5, label="data2")

plt.xlabel("Social Pillar Score", size=14)
plt.ylabel("Count", size=14)
plt.legend(loc='upper right')

# %%
