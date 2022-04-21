

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# address = 'C:\Users\btl_0\Desktop\PROJE B\Göğüs kanseri\wpbc.data\data.csv'
veri = pd.read_csv('data_1.csv')
veri.head()

veri= veri.drop(["filename"] ,axis = 1) #inplace=True
# veri.describe().T


sns.countplot(veri["label"])
print(veri.label.value_counts())


# veri["diagnosis"] =[1 if i.strip() == "M" else 0 for i in veri.diagnosis]

y= veri["label"]
X = veri.drop(columns =["label"])
# y.describe().T
# Prediction
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##KNN
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

nkomsuluk=[1,2,3,4,5,6,7,8,9,10]
uzaklik=["minkowski","manhattan","euclidean",'chebyshev']

eniyi_score=0
for j in uzaklik:
     for i in nkomsuluk:
        
         knn=KNeighborsClassifier(n_neighbors=i, metric=j)
         knn.fit(X_train,y_train)
        
         y_pred=knn.predict(X_test)
         acc=accuracy_score(y_test, y_pred)#doğruluk
         print("\nsonuclar: " ,
                   "\nn_komsuluk: ",i ,
                   "\nuzaklık: ",j,
                   "\nKNN acc: ",acc,
                   "\n------------------------")
         if eniyi_score<acc:
            eniyi_score=acc
            bi=i
            bj=j
            
knn=KNeighborsClassifier(n_neighbors=bi, metric=bj)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
#                          columns=['predicted_cancer','predicted_healthy'])
# confusion
# sns.heatmap(confusion, annot=True)

rapor=classification_report(y_test,y_pred)

basari2 = cross_val_score(estimator = knn, X=X_train, y=y_train , cv = 6)
print("basari2:",basari2.mean())
print("\nsonuclar: " ,
      "\nn_komsuluk: ",bi ,
      "\nuzaklık: ",bj,
      "\nKNN acc: ",eniyi_score)
print("\nCM:" ,cm)
print("sınıflandırma raporu:")
print(rapor )


sns.heatmap(cm, annot=True)


# SVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cekirdek =["linear", "poly", "rbf", "sigmoid" ]
eniyi_score = 0

for i in cekirdek:
    classifier = SVC(kernel = i, random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print("\nsonuclar:",
          "\nkernel:",i,
          "\nacc'si:",acc,
          "\n-----------------------------")
    if eniyi_score<acc:
       eniyi_score = acc
       bi =i


classifier = SVC(kernel = bi, random_state = 0)
classifier.fit(X_train, y_train)

y_pred= classifier.predict(X_test)

best_score=accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
rapor=classification_report(y_test,y_pred)   

from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print("basari1: ", basari.mean())
print("std: ", basari.std()) #standart sapması

print("CM:" ,cm)
print("\nsonuclar:",
      "\nkernel:",bi,
      "\nacc:" , eniyi_score)
print("rapor:",rapor)


sns.heatmap(cm, annot=True)





























