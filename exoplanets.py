import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import preprocessing

#Load, clean and display data
planets = pd.read_csv("stardata.csv", sep=',', header=0, low_memory=False, names =['pl_pnum','st_optmag','gaia_gmag','st_teff', 'st_mass','st_rad'])
planets.dropna(inplace=True)
#planets.plot(kind='box', subplots=True, sharex=False, sharey=False)
planets.hist()
plt.show()
print(planets.describe())

#Split validation dataset
array = planets.values
X = array[:, :-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
Y_encoded = preprocessing.LabelEncoder().fit_transform(Y_train)

#Compare Algorithm Performance
results = []
names = []
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X_train, Y_encoded, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Compare Model Results
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()