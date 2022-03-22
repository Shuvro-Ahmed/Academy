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
#Train_Test Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ht_fail.iloc[:, :-1], ht_fail.iloc[:,-1],random_state=1)
#SVM
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x_train, y_train)
pre_score_svm = svc.score(x_test, y_test)
print("Training accuracy of the model is {:.2f}".format(svc.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(x_test, y_test)))
predictions = svc.predict(x_test)
print(predictions)
#MLP
from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=1000000)
nnc.fit(x_train, y_train)
pre_score_mlp = nnc.score(x_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(x_test, y_test)))
predictions = nnc.predict(x_test)
print(predictions)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)
pre_score_rndmForest = rfc.score(x_test, y_test)
print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))

predictions = rfc.predict(x_test)
print(predictions)
#performance without dimensionality reduction
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
print("Training accuracy is {:.2f}".format(knn.score(x_train, y_train)) )
print("Testing accuracy is {:.2f} ".format(knn.score(x_test, y_test)) )
htfail_origin = np.array(ht_fail.iloc[:, :-1])
htfail_origin_target = np.array(ht_fail.iloc[:,-1])
#dimensionality reduction
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
htfail_df= pd.DataFrame(scaler.fit_transform(htfail_origin.data))
htfail_df=htfail_df.assign(target=htfail_origin_target)

#PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components=7)
principal_components= pca.fit_transform(htfail_origin.data)
print(principal_components)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
principal_df = pd.DataFrame(data=principal_components)
main_df=pd.concat([principal_df, htfail_df[["target"]]], axis=1)
main_df.head()
#Accuracy
x = main_df.drop('target', axis=1)
y = main_df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
print("Training accuracy of the model is  {:.2f}".format(knn.score(x_train, y_train)))
print("Testing accuracy of the model is  {:.2f}".format(knn.score(x_test, y_test)))
svc.fit(x_train, y_train)
post_score_svc = svc.score(x_test, y_test)
print("Training accuracy of the model is  {:.2f}".format(svc.score(x_train, y_train)))
print("Testing accuracy of the model is  {:.2f}".format(svc.score(x_test, y_test)))
nnc.fit(x_train, y_train)
post_score_nnc = nnc.score(x_test, y_test)
print("Training accuracy of the model is  {:.2f}".format(nnc.score(x_train, y_train)))
print("Testing accuracy of the model is  {:.2f}".format(nnc.score(x_test, y_test)))
rfc.fit(x_train, y_train)
post_score_rfc = rfc.score(x_test, y_test)
print("Training accuracy of the model is  {:.2f}".format(rfc.score(x_train, y_train)))
print("Testing accuracy of the model is  {:.2f}".format(rfc.score(x_test, y_test)))
#Bar chart
import matplotlib.pyplot as plt
scr = [pre_score_svm, post_score_svc, pre_score_mlp, post_score_nnc, pre_score_rndmForest, post_score_rfc]
title = ['SVM Pre-PCA', 'SVM Post-PCA', 'NNC Pre-PCA', 'NNC Post-PCA ', 'RFC Pre-PCA', 'RFC Post-PCA']
plt.bar(title, scr, color='#0C090A')
plt.title('Frequency of Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
plt.show()

