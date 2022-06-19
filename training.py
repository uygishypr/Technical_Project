#%%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
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
from sklearn.impute import KNNImputer
from numpy import mean
from numpy import std
from matplotlib import pyplot
import xgboost as xgb


#%%
df_merged_sorted = pd.read_excel("Nans_removed_dataset.xlsx", index_col =  [0, 1])
companies_list = df_merged_sorted.index.get_level_values('Company Name').unique().tolist()
print(companies_list)

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
vertical_concat.head()
#%%
vertical_concat_features = vertical_concat.iloc[:,2:56]
# N_neighbors is a preprocessing hyperparameter
impute_knn = KNNImputer(n_neighbors=5)
knn_imputed_df = impute_knn.fit_transform(vertical_concat_features)

#%% TRAINING
# Drop the row that has NaN values (last row)
X = np.array(vertical_concat_features)
y = np.array(vertical_concat["Label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.3, random_state=42)

#%%
results = list()
strategies = [str(i) for i in [1,3,5,7,9,15,18,21]]

for s in strategies:
	# create the modeling pipeline
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring='precision', cv=cv, n_jobs=-1, error_score='raise')
    results.append(scores)
    print('Mean Precision: %.3f (%.3f)' % (mean(scores), std(scores)))

pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()


#%%
from xgboost import XGBClassifier

xgboost_model = XGBClassifier()
xgboost_model.fit(X_train, y_train)

y_pred = xgboost_model.predict(X_test)

predictions = [round(value) for value in y_pred]

precision = precision_score(y_test, predictions)

#%%
precision

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
