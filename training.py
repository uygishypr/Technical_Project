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
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import mean, nanargmin
from numpy import std
from matplotlib import pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv
import shap

#%%
df_merged_sorted = pd.read_excel("Nans_removed_dataset.xlsx", index_col =  [0, 1])
companies_list = df_merged_sorted.index.get_level_values('Company Name').unique().tolist()
print(companies_list)
#%%
# Check the number of missing values before filling
df_merged_sorted.isna().sum()
columns_list = list(df_merged_sorted.drop(columns=['Sum','Label','GICS Sector Name','Identifier (RIC)']))

df_list = []

#%%

columns_list

#%%
plt.rcParams["figure.dpi"] = 600
nans_before_bfill = df_merged_sorted.isna().sum()
ax1 = nans_before_bfill.plot.bar(x = list(nans_before_bfill.index),y =  nans_before_bfill.values.flatten().tolist(),figsize = (10,3))
ax1.set(ylabel= "Number of missing values before backward filling")


#%%
for company in companies_list:
    subset_df = df_merged_sorted.loc[company].bfill(axis = "rows")
    df_list.append(subset_df)

vertical_concat = pd.concat(df_list, axis=0)

#%%
nans_after_bfill = vertical_concat.isna().sum() 
nans_after_bfill.index.tolist()
#%%
plt.rcParams["figure.dpi"] = 600
ax2 = nans_after_bfill.plot.bar(x = list(nans_after_bfill.index),y =  nans_after_bfill.values.flatten().tolist(),figsize = (10,3))
ax2.set(ylabel= "Number of missing values after backward filling")

#%%
vertical_concat['GICS Sector Name'].unique()
vertical_concat_features = vertical_concat.iloc[:,2:56]

#%%
# N_neighbors is a preprocessing hyperparameter
impute_knn = KNNImputer(n_neighbors=9)
knn_imputed_df = impute_knn.fit_transform(vertical_concat_features)

#%%
ohe = OneHotEncoder(sparse=False)
ohe_transformed = ohe.fit_transform(vertical_concat_features)

#%% TRAINING
# Changed X from KNN imputer to One Hot Encoded
X = np.array(knn_imputed_df)
y = np.array(vertical_concat["Label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.2,random_state=42, stratify=y)
#%%
X_ohe = np.array(ohe_transformed)
y_ohe = np.array(vertical_concat["Label"])
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X, y, 
                                   test_size=0.2,random_state=42, stratify=y)

#%%
_,_,_,df_predictions = train_test_split(X, vertical_concat, 
                                   test_size=0.2,random_state=42, stratify=y)
df_predictions.shape
#%%
## RANDOM FOREST
rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(X_train,y_train)
rf_predictions = rnd_clf.predict(X_test)
rf_importances = rnd_clf.feature_importances_
indices = np.argsort(rf_importances)[::-1]

#%%
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_predictions))
print("Random Forest Precision: ", precision_score(y_test, rf_predictions))
print("Random Forest Recall: ", recall_score(y_test, rf_predictions))
print("Random Forest F1 Score: ", f1_score(y_test, rf_predictions))

#%%
## RF CROSS VAL
kfold = StratifiedKFold(n_splits=5)
rf_results = cross_val_score(rnd_clf, X, y, cv=kfold, scoring = "recall")

#%%
## SUPPORT VECTOR MACHINE 
svm_clf = SVC(kernel='rbf',probability=True)
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)
#%%
print("SVM Accuracy: ", accuracy_score(y_test, svm_predictions))
print("SVM Precision: ", precision_score(y_test, svm_predictions))
print("SVM Recall: ", recall_score(y_test, svm_predictions))
print("SVM F1 Score: ", f1_score(y_test, svm_predictions))

#%%
## SVM CROSS VAL
kfold = StratifiedKFold(n_splits=5)
svm_results = cross_val_score(svm_clf, X, y, cv=kfold, scoring = "precision")
svm_results
#%%
# plotting a line plot after changing it's width and height
plt.figure(figsize = (8,4), dpi=600)

plt.ylabel('Feature Importance')
plt.bar(range(X_train.shape[1]),
        rf_importances[indices],
        align = "center")

feature_labels = vertical_concat_features.columns
plt.xticks(range(X_train.shape[1]),
                feature_labels[indices],rotation = 90)

plt.xlim([-1, X_train.shape[1]])

## XGBOOST 
#%%
data_dmatrix = xgb.DMatrix(data=X,label=y)

params = {
            'objective' : 'binary:logistic',
            'verbosity': 1,
            'reg_alpha': 0, 
            'early_stopping_rounds':10                                      
         }

xgboost_model = XGBClassifier(**params)
xgboost_model.fit(X_train, y_train, verbose = True)
xgboost_predictions = xgboost_model.predict(X_test)

#%%
print("XGBoost Accuracy Score: ", accuracy_score(y_test, xgboost_predictions))
print("XGBoost Precision Score: ", precision_score(y_test, xgboost_predictions))
print("XGBoost Recall Score: ", recall_score(y_test, xgboost_predictions))
print("XGBoost F1 Score: ", f1_score(y_test, xgboost_predictions))

#%%
## XGBOOST CROSS VALIDATION
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,
                    early_stopping_rounds=15,metrics="auc",
                    as_pandas=True, seed=42)
cv_results.head()

#%%
plot_confusion_matrix(xgboost_model, X_test, y_test)
plot_roc_curve(xgboost_model, X_test, y_test)
plt.show() 
#%%
plot_confusion_matrix(rnd_clf, X_test, y_test)
plot_roc_curve(rnd_clf, X_test, y_test)
plt.show() 
#%%
plot_confusion_matrix(svm_clf, X_test, y_test)
plot_roc_curve(svm_clf, X_test, y_test)
plt.show() 
#%%
#%%
rf_probability_predictions = rnd_clf.predict_proba(X_test)[:,1]
svm_probability_predictions = svm_clf.predict_proba(X_test)[:,1]
xgb_probability_predictions = xgboost_model.predict_proba(X_test)[:,1]

#%%
#df_predictions.drop(columns=['RF Controversy Predictions'], inplace = True)
#%%
df_predictions['RF Controversy Predictions'] = rf_probability_predictions.tolist()
df_predictions['XGB Controversy Predictions'] = xgb_probability_predictions.tolist()
df_predictions['SVM Controversy Predictions'] = svm_probability_predictions.tolist()

#%%
df_predictions.head()
#%%
grouped_predictions = df_predictions.groupby(["Identifier (RIC)"])
#%%
identifier_list = []
sums = []
rf_ratings = []
xgb_ratings = []
svm_ratings = []

#%%
for key, item in grouped_predictions:

    group = grouped_predictions.get_group(key)
    rf_average_proba = group["RF Controversy Predictions"].mean()
    xgb_average_proba = group["XGB Controversy Predictions"].mean()
    svm_average_proba = group["SVM Controversy Predictions"].mean()
    rf_rating = 100 * (1 - rf_average_proba)
    xgb_rating = 100 * (1 - xgb_average_proba)
    svm_rating = 100 * (1 - svm_average_proba)
    sum_controversies_for_company = group["Sum"].sum()
    sums.append(sum_controversies_for_company)
    rf_ratings.append(rf_rating)
    xgb_ratings.append(xgb_rating)
    svm_ratings.append(svm_rating)

#%%
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(sums, rf_ratings, marker="+")
axs[0, 0].set_title('Random Forest')
axs[0, 1].scatter(sums, xgb_ratings, marker="+")
axs[0, 1].set_title('XGBoost')
axs[1, 0].scatter(sums, svm_ratings, marker="+")
axs[1, 0].set_title('SVM RBF')

#plt.xlabel("Number of Controversies")
#plt.ylabel("Social Controversy Rating")
plt.show()
#%%
from scipy import stats
corr_x = sums
corr_y = xgb_ratings
pearson_corr = stats.pearsonr(corr_x,corr_y)
pearson_corr

#%%
# DID THIS IN COLAB , TOO LONG
results = list()
strategies = [str(i) for i in [1,3,5,7,9,15,18,21]]

for s in strategies:
	# create the modeling pipeline
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring='precision', cv=cv, n_jobs=-1, error_score='raise')
    results.append(scores)
    print('Mean Precision: %.3f (%.3f)' % (mean(scores), std(scores)))

plt.boxplot(results, labels=strategies, showmeans=True)
plt.show()

#%%
from collections import Counter
from imblearn.over_sampling import SMOTE

# SMOTE DOES NOT WORK TRY BOOTSTRAPPING
sm = SMOTE(random_state = 42) 
X_res, y_res = sm.fit_resample(X_train, y_train)

print('Original dataset shape %s'%Counter(y_train))
print('Resampled dataset shape %s'%Counter(y_res))

#%%
def confusion_matrix_eval(Y_truth, Y_pred):
    cm = confusion_matrix(Y_truth, Y_pred)
    cr = classification_report(Y_truth, Y_pred)
    print("-"*90)
    print("[CLASS_REPORT] printing classification report to console")
    print("-"*90)
    print(cr)
    print("-"*90)
    return [cm, cr]


#%%
explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_train)

#%%
plt_shap = shap.summary_plot(shap_values, #Use Shap values array
                             features=X_train, # Use training set features
                             feature_names=columns_list, #Use column names
                             show=False, #Set to false to output to folder
                             plot_size=[8,6]) # Change plot size


#%%
## XGBOOST VISUALIZATION ATTEMPTS 
booster = xgboost_model.get_booster()
importance = booster.get_score(importance_type="gain")
ax = xgb.plot_importance(importance, max_num_features=20, importance_type='gain', show_values=True)
#%%
xgb.plot_importance(xgboost_model, max_num_features=20)
plt.show()
#%%
plt.figure(figsize = (6,15), dpi = 600)
plt.xticks(rotation=90)
plt.show()
#%%
features = list(vertical_concat_features.columns)
#%%
sorted_idx = xgboost_model.feature_importances_.argsort()
#%%
plt.barh(features[sorted_idx], xgboost_model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")
#%%
plt.figure(figsize = (3,6), dpi = 600)
plt.xticks(rotation=90)
plt.bar(range(len(xgboost_model.feature_importances_)), xgboost_model.feature_importances_)
plt.show()

#%%
## NN structure with 54 inputs and 1 output 
from sklearn.preprocessing import LabelEncoder
from keras import metrics

#%%
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(54, input_dim=54, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.AUC(),
                                                                   metrics.Precision(),
                                                                   metrics.Recall()])
	return model

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=2)

#%%
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

#%%
results

#%%    
print(classification_report(y_true, y_pred, target_names=target_names))
#%%
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
dt_predictions = tree_clf.predict(X_test)



#%%
plt.scatter(rf_probability_predictions[:,0])
#%%
plt.figure(figsize=(15,7))
plt.hist(rf_probability_predictions[:,0], bins = 50, label='Positives', alpha=0.7, color='r')
plt.xlabel('Probability of Controversy', fontsize=25)
plt.ylabel('Number of records in each bucket', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
#%%
voting_clf = VotingClassifier(
    estimators = [('Decision Trees',tree_clf), ('Random Forests',rnd_clf),('SVC',svm_clf)],
    voting = 'hard'
)
voting_clf.fit(X_train,y_train)

for clf in (svm_clf, tree_clf, rnd_clf, voting_clf):
    clf.fit(X_train,y_train)
    y_predictions = clf.predict(X_test)
    print(clf.__class__.__name__,precision_score(y_test,y_predictions))

#%%
plot_confusion_matrix(voting_clf, X_test, y_test)
plt.show() 