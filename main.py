import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn import tree
import matplotlib.pyplot as plt



# This function counts the number of missing values in a column
def count_missing(data):
    total = len(data)
    missing_v = 0
    for item in data:
        if item == '?':
            missing_v += 1

    return (missing_v, total)


# This function will count the number of extremes and outliers for the given data.
# Outlier: exceeds 3 x std, within5 x std.
# Extreme: exceeds 5 x std.
def count_extreme_outlier_no(data):
    data = data.to_numpy()
    no_extreme = 0
    no_outlier = 0
    std = np.std(data)
    # print(std)
    mean = np.mean(data)
    # print(mean)
    outlier_extreme_id = []

    upper_outlier_threshold = mean + 3 * std
    upper_extreme_threshold = mean + 5 * std
    lower_outlier_threshold = mean - 3 * std
    lower_extreme_threshold = mean - 5 * std

    for i in range(len(data)):
        if data[i] > upper_outlier_threshold:
            outlier_extreme_id.append(i)
            if data[i] > upper_extreme_threshold:
                no_extreme += 1
                continue
            no_outlier += 1

        if data[i] < lower_outlier_threshold:
            outlier_extreme_id.append(i)
            if data[i] < lower_extreme_threshold:
                no_extreme += 1
                continue
            no_outlier += 1

    return (no_extreme, no_outlier, outlier_extreme_id)


############  Load Data  ############

df1 = pd.read_csv('infomation.csv')
# print(df1.info())

df2 = pd.read_csv('finance.csv')
# print(df2.info())

# Merge two files on 'ID'
df = pd.merge(df1, df2, on='ID')
# print(df.info())

# df = pd.read_csv('adult.csv')

'''
marital_status_num = to_num(df['Marital status'])
df['Marital status num'] = marital_status_num
matrix1 = df[['Income', 'Age']].groupby('Age')
matrix1_desc = matrix1.describe()
'''

# print(df['Income'].describe())


############  Data Exploration  ############

# Format 'Income'
# 0 represents <= 85K
# 1 represents > 85K
income_encoder = LabelEncoder()
income_num = income_encoder.fit_transform(df['Income'])
df['Income'] = income_num
# print(df['Income'].value_counts())
#print(income_encoder.inverse_transform([0, 1]))

# Evaluate the correlation between Income and Work hours per week
correlation = df[['Income', 'Work hours per week']].corr()
# print(correlation)

# Plot the outliers in 'Capital gain'
# print(df['Capital loss'].describe())
# df['Capital loss'].plot.hist()
# plt.show()

############  Data Preparation  ############

# Count missing values in every column
for column in df:
    missing_no = count_missing(df[column])
    # print(f'Attribute: {column}, {missing_no[0]} missing values out of {missing_no[1]} records.')

# Process missing values: Drop all rows containing missing values
row_id = []
for attribute in ['Workclass', 'Occupation', 'Native country']:
    id = 0
    for value in df[attribute]:
        if value == '?' and id not in row_id:
            row_id.append(id)
        id += 1

df_no_missing = df.drop(row_id, axis=0)

for column in df_no_missing:
    missing_no = count_missing(df_no_missing[column])
    # print(f'Attribute: {column}, {missing_no[0]} missing values out of {missing_no[1]} records.')

# Count the number of outliers and extremes in 'Work hours per week', 'Capital gain' and 'Capital loss'
for attribute in ['Work hours per week', 'Capital gain', 'Capital loss']:
    extreme, outlier, id = count_extreme_outlier_no(df_no_missing[attribute])
    # print(f"In '{attribute}':")
    # print(f' - Number of Outliers: {outlier}')
    # print(f' - Number of Extremes: {extreme}\n')

# Drop attributes 'ID', 'Race', 'Sex'
df_prepared = df_no_missing.drop(['ID', 'Race', 'Sex'], axis=1)
# print(df_prepared.info())


# Derived attribute 'Net capital gain'
cap_gain = df_prepared['Capital gain'].to_numpy()
cap_loss = df_prepared['Capital loss'].to_numpy()
net_cap_gain = []
for i in range(len(cap_gain)):
    net_value = cap_gain[i] - cap_loss[i]
    net_cap_gain.append(net_value)
df_prepared['Net capital gain'] = net_cap_gain
# print(df_prepared['Net capital gain'])


# Age group
age_group = []
for age in df_prepared['Age']:
    if age >= 18 and age < 27:
        age_group.append('18-26')
    elif age >= 27 and age < 37:
        age_group.append('27-36')
    elif age >= 37 and age < 47:
        age_group.append('37-46')
    elif age >= 47 and age < 57:
        age_group.append('47-56')
    else:
        age_group.append('57+')
df_prepared['Age group'] = age_group
# print(df_prepared['Age group'])


# Work intensity
work_intensity = []
for hours in df_prepared['Work hours per week']:
    if hours < 20:
        work_intensity.append('Very Low')
    elif hours >= 20 and hours < 36:
        work_intensity.append('Low')
    elif hours >= 36 and hours < 45:
        work_intensity.append('Normal')
    elif hours >= 45 and hours < 61:
        work_intensity.append('High')
    else:
        work_intensity.append('Very High')
df_prepared['Work intensity'] = work_intensity
# print(df_prepared['Work intensity'])


# Balancing
# print('Before Boosting:')
# print(df_prepared['Income'].value_counts())
df_high = df_prepared[df['Income'] == 0]
df_low = df_prepared[df['Income'] == 1]
df_low_boost = resample(df_low, replace=True, n_samples=len(df_high))
df_boosted = pd.concat([df_high, df_low_boost])
# print('\nAfter Boosting:')
# print(df_boosted['Income'].value_counts())

# Format the data as required
#print(df_boosted.info())
df_ohe = pd.get_dummies(df_boosted, columns=['Workclass', 'Marital status', 'Occupation', 'Relationship',
                                               'Native country', 'Age group', 'Work intensity'])
#print(df_ohe.info())

# print(df_boosted['Native country'].value_counts())
# df_boosted['Native country'].value_counts().plot.pie()
# plt.show()

# Dropping all 'Native country' related attributes
nativecountry_rel_index = []
for n in range(40, 81):
    nativecountry_rel_index.append(n)

df_ohe.drop(df_ohe.columns[nativecountry_rel_index], axis=1, inplace=True)

# Transform 'Capital gain'
# df_ohe['Capital gain'].value_counts().plot.hist()
# plt.show()
has_capital_gain = []
for value in df_ohe['Capital gain']:
    if value == 0:
        has_capital_gain.append(0)
    else:
        has_capital_gain.append(1)
df_ohe['Have capital gain'] = has_capital_gain

# Transform 'Capital loss'
# df_ohe['Capital loss'].value_counts().plot.hist()
# plt.show()
has_capital_loss = []
for value in df_ohe['Capital loss']:
    if value == 0:
        has_capital_loss.append(0)
    else:
        has_capital_loss.append(1)
df_ohe['Have capital loss'] = has_capital_loss

# Transform 'Net capital gain'
# df_boosted['Net capital gain'].value_counts().plot.hist()
# plt.show()
net_sign = []
for value in df_ohe['Net capital gain']:
    if value >= 0:
        net_sign.append(1)
    else:
        net_sign.append(0)
df_ohe['Net capital gain sign'] = net_sign

#print(df_ohe.info())

# Removing data redundancy
df_ohe = df_ohe.drop('Work hours per week', axis=1)
df_ohe = df_ohe.drop('Age', axis=1)
df_ohe = df_ohe.drop(['Capital gain', 'Capital loss'], axis=1)
df_ohe = df_ohe.drop('Net capital gain', axis=1)
df_ohe = df_ohe.drop(['Education'], axis=1)
#print(df_ohe.info())


# split training set & test set
X = df_ohe.drop('Income', axis=1)
y = df_ohe['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Decision Tree
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_train_predict = dt_model.predict(X_train)

train_accuracy = round(accuracy_score(dt_train_predict, y_train), 4)
test_accuracy = round(accuracy_score(dt_predict, y_test), 4)
print('-- Decision Tree Model Accuracy --')
print(f'Train Accuracy = {train_accuracy}')
print(f'Test Accuracy = {test_accuracy}\n')


# Decision tree Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [2, 6, 10],
    'min_samples_leaf': [1, 3, 5]
}

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best hyperparameters: {best_params}")

best_dt_model = grid_search.best_estimator_

# Predictions
dt_predict = best_dt_model.predict(X_test)
dt_train_predict = best_dt_model.predict(X_train)

# Calculate accuracies
train_accuracy = round(accuracy_score(dt_train_predict, y_train), 4)
test_accuracy = round(accuracy_score(dt_predict, y_test), 4)
print('-- Decision Tree Model (Tuned) Accuracy --')
print(f'Train Accuracy = {train_accuracy}')
print(f'Test Accuracy = {test_accuracy}\n')




# Naive bayes
nb_model = GaussianNB().fit(X_train, y_train)
nb_predict = nb_model.predict(X_test)
nb_train_predict = nb_model.predict(X_train)
print('-- Naive Bayes Model Accuracy --')
train_accuracy = round(accuracy_score(y_train, nb_train_predict), 4)
print(f'Train accuracy = {train_accuracy}')

test_accuracy = round(accuracy_score(y_test, nb_predict), 4)
print(f'Test accuracy = {test_accuracy}\n')






# Neural network
nn_model = MLPClassifier().fit(X_train, y_train)
nn_predict = nn_model.predict(X_test)
nn_train_predict = nn_model.predict(X_train)
print('-- Neural Network Model Accuracy --')

train_accuracy = round(accuracy_score(y_train, nn_train_predict), 4)
print(f'Train Accuracy = {train_accuracy}')
test_accuracy = round(accuracy_score(y_test, nn_predict), 4)
print(f'Test Accuracy = {test_accuracy}')





# Evaluation
train_accuracy = round(accuracy_score(dt_train_predict, y_train), 4)
test_accuracy = round(accuracy_score(dt_predict, y_test), 4)
f1_score = round(f1_score(dt_predict, y_test), 4)
roc_auc = round(roc_auc_score(dt_predict, y_test), 4)


print('-- Decision Tree Model Evaluation --')
print(f'Train Accuracy = {train_accuracy}')
print(f'Test Accuracy = {test_accuracy}')
print(f'F1 Score = {f1_score}')
print(f'ROC AUC Score = {roc_auc}')





# Patterns
feature_names = X.columns
#print(feature_names)
class_names = ['<=85K','>85K']

text_representation = tree.export_text(dt_model, feature_names=feature_names, class_names=class_names)

print('\n--------- Decision Tree Model ---------')
print(text_representation)




# Visualisation

# Visualise the Tree
from sklearn.tree import plot_tree, export_graphviz

# Using plot_tree
plot_tree(dt_model)


# Visualise pattern 1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filter data based on given conditions
filtered_data = df_ohe[(df_ohe['Marital status_Married-civ-spouse'] <= 0.5) &
                   (df_ohe['Education num'] <= 12.5) &
                   (df_ohe['Have capital gain'] <= 0.5)]

plt.figure(figsize=(10,6))
sns.countplot(x='Marital status_Never-married', hue='Income', data=filtered_data)
plt.title('Income Distribution based on Marital Status of Never-married for Filtered Individuals')
plt.show()

# Visualise pattern 2
# Filter data based on conditions
filtered_data = df_ohe[(df_ohe['Marital status_Married-civ-spouse'] <= 0.5) &
                   (df_ohe['Education num'] > 12.5) &
                   (df_ohe['Have capital gain'] > 0.5)]
# Plot
plt.figure(figsize=(10,6))
sns.countplot(x='Age group_18-26', hue='Income', data=filtered_data)
plt.title('Income Distribution based on Age Group (18-26) for Educated Singles with Capital Gains')
plt.show()




# Visualise pattern 3

# Filter data based on conditions
filtered_data = df_ohe[(df_ohe['Marital status_Married-civ-spouse'] > 0.5) &
                   (df_ohe['Education num'] > 11.5) &
                   (df_ohe['Have capital gain'] <= 0.5)]

# Plot
plt.figure(figsize=(10,6))
sns.countplot(x='Net capital gain sign', hue='Income', data=filtered_data)
plt.title('Income Distribution based on Net Capital Gain Sign for Highly Educated Individuals Married to a Civilian Spouse')
plt.show()





# Parameters setting
def best_depth(X_train, X_test, y_train, y_test, n):
    i = 1
    while i <= n:
        dt_model = DecisionTreeClassifier(max_depth=i).fit(X_train, y_train)
        dt_predict = dt_model.predict(X_test)
        dt_train_predict = dt_model.predict(X_train)

        train_accuracy = round(accuracy_score(dt_train_predict, y_train), 4)
        test_accuracy = round(accuracy_score(dt_predict, y_test), 4)
        print(f'-- Decision Tree Model Max Depth = {i} --')
        print(f'Train Accuracy = {train_accuracy}')
        print(f'Test Accuracy = {test_accuracy}')
        print(f'Train Test Difference = {round(train_accuracy - test_accuracy, 3)}\n')
        i += 1

best_depth(X_train, X_test, y_train, y_test, 20)





























