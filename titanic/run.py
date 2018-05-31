
# coding: utf-8

# In[ ]:


# My first Kaggle competition


# In[23]:


# Import packages
import pandas as pd
import numpy as np
import csv
import re

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from collections import Counter
from sklearn.metrics import accuracy_score

#Classifiers

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


def load_dataset():
    # Process the dataset
    train_data =  pd.read_csv('dataset/train.csv', delimiter=',')
    #print(train_data)
    test_data =  pd.read_csv('dataset/test.csv', delimiter=',')
    #print(test_data)

    return train_data, test_data

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

def process_dataset(train_data, test_data):
    full_data = [train_data, test_data]

    # detect outliers from Age, SibSp , Parch and Fare
    Outliers_to_drop = detect_outliers(train_data, 2, ["Age", "SibSp", "Parch", "Fare"])

    # Gives the length of the name
    train_data['Name_length'] = train_data['Name'].apply(len)
    test_data['Name_length'] = test_data['Name'].apply(len)

    # Feature that tells whether a passenger had a cabin on the Titanic
    train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())
    
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    

    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
        # Mapping Fare
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    
        # Mapping Age
        dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

    return train_data, test_data

def draw_pearson_table(train_data):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)

def get_classifiers():
    classifiers_dict = {}
    classifiers_dict['ensemble'] = [BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier]
    classifiers_dict['tree'] = [DecisionTreeClassifier]
    classifiers_dict['svm'] = [SVC, LinearSVC, NuSVC]

    return classifiers_dict


def get_classifier(type, name):
    if type == 'ensemble':
        return
    elif type == 'tree':
        return
    elif type == 'svm':
        return
    else:
        return svm.SVC()

def train_model(train_data, train_label, clf):
    clf.fit(train_data, train_label)
    return clf

def write_prediction(pid, pred):
    pred = np.apply_along_axis(lambda y: [str(i) for i in y], 0, pred).tolist()
    a = "Survived"
    pred = [a] + pred

    myFile = open('gender_submission.csv', 'w')
    with myFile:
        i = 0
        while (i < len(pid)):
            myData = pid[i] + ',' + pred[i] + "\n"

            myFile.write(myData)
            # print(myData)
            i += 1


def main():
    train_data, test_data = load_dataset()
    train_data, test_data = process_dataset(train_data, test_data)

    train_label = train_data[['Survived']]

    pid = test_data[['PassengerId']]
    pid = pid.astype(str)
    pid = pid['PassengerId'].values.tolist()
    pid = ['PassengerId'] + pid

    delete_columns = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']

    train_data.drop(delete_columns, axis=1, inplace=True)

    delete_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    test_data.drop(delete_columns, axis=1, inplace=True)

    draw_pearson_table(train_data)

    clf = get_classifier()
    clf = train_model(train_data, train_label, clf)

    pred = clf.predict(test_data)

    write_prediction(pid, pred)

main()

