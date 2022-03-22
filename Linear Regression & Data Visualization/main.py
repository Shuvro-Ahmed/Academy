import pandas as pd
import numpy as np
ht_fail = pd.read_csv('/content/sample_data/heart failur classification dataset.csv')
ht_fail.head(5)
ht_fail.shape
ht_fail.isnull()
ht_fail.isnull().sum()
#Imputing missing values
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(ht_fail[['time']])

ht_fail['time'] = impute.transform(ht_fail[['time']])
ht_fail[['time']]
#Imputing missing values
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(ht_fail[['serum_sodium']])

ht_fail['serum_sodium'] = impute.transform(ht_fail[['serum_sodium']])
ht_fail[['serum_sodium']]
ht_fail.isnull().sum()
#Handling categorical features
#ht_fail.info
ht_fail
ht_fail['smoking'].unique()
ht_fail['smoking'] = ht_fail['smoking'].map({'No':0,'Yes':1}) 
ht_fail
ht_fail['sex'].unique()
ht_fail['sex'] = ht_fail['sex'].map({'Male':0,'Female':1}) 
ht_fail
#Scaling/ MinMax scaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ht_fail.iloc[:, :-1], ht_fail.iloc[:,-1],random_state=1)
print(X_train.shape)
print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)
Scaled_X_train = scaler.transform(X_train)
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    Scaled_X_train.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    Scaled_X_train.max(axis=0)))
#Spliting the dataset into features and labels
ht_fail = ht_fail.drop('Unnamed: 0', axis=1)
features = ht_fail.iloc[:, :-1]
print(features)

lables = ht_fail.iloc[:,-1]
print(lables)
#correlation & heatmap
ht_failCorr = ht_fail.corr()
ht_failCorr
import seaborn as sea
sea.heatmap(ht_failCorr,cmap='YlGnBu')
#logistic regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
ht_fail.head()
#training set
#Prepare the training set

X = ht_fail.iloc[:, :-1]

y = ht_fail.iloc[:, -1]
# 8:2 train-test split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Training
model = LogisticRegression()
model.fit(x_train, y_train) 
predictions = model.predict(x_test)
print(predictions)
#Accuracy_score
log_accuracy = accuracy_score(y_test, predictions)
print(log_accuracy)
#decision tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = ht_fail.iloc[:,0:12]
Y = ht_fail.iloc[:,12]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
dec_accuracy = accuracy_score(y_pred,y_test)
print(dec_accuracy)
from sklearn import tree
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(clf, feature_names = X.columns, class_names=['1','2','3','4','5','6','7','8'], filled = True);
#Bar Chart
t_accuracy = [log_accuracy, dec_accuracy]
temp = ['Logistic Regression',"Decision Tree"]
plt.bar(temp, t_accuracy, color='#0C090A')
plt.title('Frequency of Accuracy')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()

