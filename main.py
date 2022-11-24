# Step 1 - Problem formulation :
# Supervised learning problem : given a dataset with 17 features predict if the patient has a heart disease or not

# Importing the necessary packages
# For data manipulation
import pandas as pd
import numpy as np
from sklearn import metrics
# For splitting data into train and test datasets
from sklearn.model_selection import train_test_split
# For data visualization
from sklearn import tree
import matplotlib.pyplot as plt
# import seaborn as sns
# For using the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# Reload plt to avoid str' object is not callable error when using plt.title()
from importlib import reload
plt=reload(plt)
# For calculating the metrics
import sklearn.metrics as skl
# For performing K-fold cross validation and calculate the score
from sklearn.model_selection import cross_val_score
# For transforming values like "yes" to 1 and "no" to 0 - encoding categorical data
from sklearn.preprocessing import LabelEncoder
# For feature scaling
from sklearn.preprocessing import StandardScaler
# For balancing the biased output variable
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE
# For finding the optimal hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# For calculating bias and variance
# from mlxtend.evaluate import bias_variance_decomp
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier

# from xgboost import XGBClassifier
# from sklearn.metrics import mean_squared_error

import xgboost as xgb


# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.over_sampling import SVMSMOTE
# from imblearn.over_sampling import KMeansSMOTE

import pickle

# Step 2 - Data collection, assessment and management
# Reading the dataset from url into pandas dataframe
features_csv_file = "/Users/aelangovan/CS688_Smart_Farming_AI_model/final_crop_recommendation.csv"
_dataset = pd.read_csv(features_csv_file)
print(_dataset.shape)

# To show missing values in the dataset
# import missingno as msno
# msno.matrix(_dataset)  # just to visualize. no missing value.

# # Data preprocessing - encoding categorical data
# # To transform values like "yes" to 1 and "no" to 0 : converting string to float so the decision tree classifier understands
label_encoder = LabelEncoder()
dataset_columns = [column for column in _dataset.columns if _dataset[column].dtype == 'object']

print('classes are ', np.unique(_dataset['label']))

for column in dataset_columns:
    _dataset[column] = label_encoder.fit_transform(_dataset[column])


def remove_high_correlation(dataset):
    """Prints highly correlated features pairs in the data frame (helpful for feature engineering)"""
    # # Data visualization using heatmap
    # plt.subplots()
    # sns.heatmap(dataset.corr())
    # plt.show()

    threshold = 0.97
    corr_df = dataset.corr() # get correlations
    correlated_features = np.where(np.abs(corr_df) > threshold) # select ones above the abs threshold
    correlated_features = [(corr_df.iloc[x,y], x, y) for x, y in zip(*correlated_features) if x != y and x < y] # avoid duplication
    s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0])) # sort by correlation value

    if s_corr_list == []:
        print("There are no highly correlated features with correlation above", threshold)
    else:
        for v, i, j in s_corr_list:
            cols = dataset.columns
            print("%s and %s = %.3f" % (corr_df.index[i], corr_df.columns[j], v))

    # Drop columns that cause high correlation with threshold being 0.95
    cor_matrix = dataset.corr().abs()
    print('cor_matrix')
    print(cor_matrix)
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    print('upper_tri')
    print(upper_tri)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    print(); print('to_drop'); print(to_drop)
    print('type ', type(dataset), type(to_drop))
    global _dataset
    _dataset = dataset.drop(to_drop, axis=1)
    print(); print('dataset.head()'); print(_dataset.head())


# remove_high_correlation(_dataset)

# Pie chart to show the dataset is imbalanced - imbalanced target variable

# print("values are ", _dataset['has_covid'].values)
# variable_0, variable_1 = _dataset['has_covid'].value_counts()
# print("values are ", variable_0, variable_1)
# colors = sns.color_palette('pastel')
# plt.pie([variable_0, variable_1], labels=['Yes', 'No'], colors=colors, autopct='%0.0f%%')
# plt.title('Percentage of people having covid vs not')
# plt.show()


from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
import pandas as pd


# Generate larger synthetic dataset based on a smaller dataset

def generate_larger_dataset(T, N, k):
    # """
    # Returns (N/100) * n_minority_samples synthetic minority samples.
    #
    # Parameters
    # ----------
    # T : array-like, shape = [n_minority_samples, n_features]
    #     Holds the minority samples
    # N : percentage of new synthetic samples:
    #     n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    # k : int. Number of nearest neighbours.
    #
    # Returns
    # -------
    # S : array, shape = [(N/100) * n_minority_samples, n_features]
    # """
    n_minority_samples, n_features = T.shape
    print('samples ', n_minority_samples, n_features)

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = N/100
    n_synthetic_samples = N * n_minority_samples
    n_synthetic_samples = int(n_synthetic_samples)
    n_features = int(n_features)
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i].reshape(1, -1), return_distance=False)
        for n in range(int(N)):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it
            while nn_index == i:
                nn_index = choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[int(n + i * N), :] = T[i,:] + gap * dif[:]

    return S


# Step 3 - Feature Engineering
# Extracting Input / Attributes / Features - include all columns except 'Covid' as 'Covid' is the output
X = np.array(_dataset.iloc[:, 0:-1])
# Extracting Output / Target / Class / Labels
Y = np.array(_dataset.iloc[:, -1])
classes = np.unique(Y)
print('classes are \n', classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2, shuffle=True, stratify=Y)

# To add synthetic data
# Display the size of the 3 datasets - train, validation and test
# print("Shape of X Train: {}".format(X_train.shape))
# print("Shape of X Test: {}".format(X_test.shape))
# print("Shape of Y Train: {}".format(Y_train.shape))
# print("Shape of Y Test: {}".format(Y_test.shape))

# print("type of X Train: {}".format(type(X_train)))
# print("type of Y Train: {}".format(type(Y_train)))
# print("Y Train: {}".format(Y_train))
#
# training_set = np.insert(X_train, 107, Y_train, axis=1)
# print("type of training_set: {}".format(type(training_set)))
# print("Shape of training_set: {}".format(training_set.shape))
# training_set = pd.DataFrame(training_set, columns = _dataset.columns)
# training_set["has_covid"] = training_set["has_covid"].astype(int)
# print(training_set.head())
#
#
# dataset1 = training_set.to_numpy()
# new_data = generate_larger_dataset(dataset1,100,10) # this is where I call the function and expect new_data to be generated with larger number of samples than original df.
# # print(new_data.shape)
# #
# training_set = pd.DataFrame(new_data, columns=_dataset.columns)
#
# print('shape is ', training_set.shape)
# print('----------------------------------------------------------------')
# print('dataset.head()'); print(training_set.head())


# # Pie chart to show the dataset is imbalanced - imbalanced target variable
# print("values are ", _dataset['label'].values)
# variable_1, variable_0 = _dataset['has_covid'].value_counts()
# print("values are ", variable_0, variable_1)
# colors = sns.color_palette('pastel')
# plt.pie([variable_1, variable_0], labels=['Yes', 'No'], colors=colors, autopct='%0.0f%%')
# plt.title('Percentage of people having covid vs not')
# plt.show()


# # Write the new synthetic dataset to csv file
# features_filename = '/Users/aelangovan/masters_thesis/synthetic_dataset.csv'
# # Write to a csv file
# training_set.to_csv(features_filename, sep='\t', index=False, encoding='utf-8', mode='a')
# print("converted to csv")

# X_train = np.array(training_set.iloc[:, 0:-1])
# # Extracting Output / Target / Class / Labels
# Y_train = np.array(training_set.iloc[:, -1])


# Remove bias by balancing the dataset using SMOTE from imblearn
# oversample_using_smote = SMOTE(sampling_strategy='minority')
# Fit and apply the transform
# X_train, Y_train = oversample_using_smote.fit_resample(X_train,Y_train)

# #Random Oversampling
# random_os = RandomOverSampler(random_state = 42)
# X_train, Y_train = random_os.fit_resample(X_train, Y_train)

# #BorderlineSMOTE
# smote_border = BorderlineSMOTE(random_state = 42, kind = 'borderline-2')
# X_train, Y_train = smote_border.fit_resample(X_train, Y_train)

# #SVM SMOTE
# smote_svm = SVMSMOTE(random_state = 42)
# X_train, Y_train = smote_svm.fit_resample(X_train, Y_train)

# #K-Means SMOTE
# smote_kmeans = KMeansSMOTE(random_state = 42)
# X_train, Y_train = smote_kmeans.fit_resample(X_train, Y_train)

# print('type is ', type(X_train))

# df = pd.DataFrame(X_train, columns = _dataset.columns[0:-1])
# df['has_covid'] = Y_train
# print(df.shape)
# print(df.head())

# # Pie chart to show the dataset is balanced
# print("values are ", df['has_covid'].values)
# variable_1, variable_0 = df['has_covid'].value_counts()
# print("values are ", variable_0, variable_1)
# colors = sns.color_palette('pastel')
# plt.pie([variable_1, variable_0], labels=['Yes', 'No'], colors=colors, autopct='%0.0f%%')
# plt.title('Percentage of people having covid vs not')
# plt.show()

# Feature scaling to standardize the independent variables of the dataset in a specific range
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print('features after standardizing: ', X_train)

def decision_tree(dataset):
    # Create the Decision Tree Classifier model and fit it on the training dataset

    # estimator = xgb.XGBClassifier(
    #     objective='multi:softmax'
    # )
    #
    # parameters = {
    #     'max_depth': range (2, 10, 1),
    #     'n_estimators': range(60, 220, 40),
    #     'learning_rate': [0.1, 0.01, 0.05]
    # }
    #
    # grid_search = RandomizedSearchCV(
    #     estimator,
    #     parameters,
    #     n_jobs=5,
    #     cv=5,
    #     verbose=True
    # )
    #
    # grid_search.fit(X, Y)
    #
    # grid_search = grid_search.best_estimator_

    clf = DecisionTreeClassifier(random_state=42)

    # Tune the hyperparameters to find the best model with optimal hyperparameters

# Best hyperparameters for crop pred: {'min_samples_split': 9, 'max_features': 4, 'max_depth': 12}
# Accuracy score for training & validation dataset: 0.9755681818181818


    param_grid = {'max_depth': np.arange(1, 20), 'max_features': np.arange(1, 6),
                  'min_samples_split': np.arange(2, 20)}
    # param_grid = {'max_depth': [5]}
    clf = RandomizedSearchCV(DecisionTreeClassifier(), param_grid, cv=5)

    # Fit models to the training & validation datasets and generate the best model
    clf.fit(X_train, Y_train)
    print('\nBest hyperparameters ', clf.best_params_)
    clf = clf.best_estimator_

    # Predict Accuracy Score for training & validation set
    print("\nAccuracy score for training & validation dataset:", clf.score(X_train, Y_train))

    # Test it on test data and generate classification report, plot confusion matrix and ROC
    # Y_pred = clf.predict(X_test)
    # print('\nClassification report is - \n', skl.classification_report(Y_test, Y_pred))
    # print('\nconfusion matrix is - \n', confusion_matrix(Y_test, Y_pred).ravel())
    # tree.plot_tree(clf, feature_names=dataset.iloc[:, 0:-1].columns, class_names=classes, filled=True, rounded=True,
    #                    fontsize=14)
    # plt.show()
    #
    # # Plot feature importance for the Decision tree model
    #
    # importance = clf.feature_importances_
    # # summarize feature importance
    # for i, imp in enumerate(importance):
    #     print('Feature: %s, Score: %.5f' % (dataset.iloc[:, 0:-1].columns[i], imp))
    # # plot feature
    # plt.figure(figsize=(25, 10))
    # fig = pyplot.bar([x for x in dataset.iloc[:, 0:-1].columns], importance)
    # # fig.set_size_inches(18.5, 10.5)
    # pyplot.show()

    # s = pickle.dumps(clf)
    # clf2 = pickle.loads(s)
    # # clf2.predict(X[0])
    # print(clf2.predict(X_test))

    #save the model in pickle format

    pickle.dump(clf, open('/Users/aelangovan/CS688_Smart_Farming_AI_model/model.pkl','wb'))

    # pickle.dump(grid_search, open("/Users/aelangovan/CS688_Smart_Farming_AI_model/best_calif.pkl", "wb"))
    # clf2 = pickle.load(open("/Users/aelangovan/CS688_Smart_Farming_AI_model/best_calif.pkl", "rb"))
    # joblib.dump(bt, '/Users/aelangovan/CS688_Smart_Farming_AI_model/final_model')
    # dump(grid_search, '/Users/aelangovan/CS688_Smart_Farming_AI_model/filename.joblib')
    # clf1 = joblib.load('/Users/aelangovan/CS688_Smart_Farming_AI_model/final_model')
    # print(clf1.predict(X_test))

# import joblib
decision_tree(_dataset)
#
# clf1 = joblib.load('/Users/aelangovan/CS688_Smart_Farming_AI_model/final_model')
# print('prediction is ', clf1.predict(X_test))




