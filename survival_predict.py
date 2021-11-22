import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv('titanic.csv')

X = df.drop(['PassengerId','Survived','Name','Ticket','Fare','Cabin','Embarked'], axis=1)
y = df['Survived']

print(y.value_counts())

print(X.isnull().sum())

X['Age'] = X['Age'].fillna(X['Age'].median())
X = pd.get_dummies(X, columns=['Sex'])

print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21, stratify=y)


model_knn = KNeighborsClassifier()
param = {'n_neighbors':np.arange(2,100), 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
gscv = GridSearchCV(model_knn, param_grid=param, scoring='accuracy', cv=5)
gscv.fit(X_train, y_train)

print(gscv.best_params_)

print(gscv.best_score_)

print(gscv.predict_proba(X_test))

print(gscv.classes_)

probe = gscv.predict_proba(X_test)
print(probe.sum())

probe_pos = probe[:,1]

df_test = pd.read_csv('titanic_test.csv')
X_new = df_test.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'], axis=1)
X_new['Age'] = X_new['Age'].fillna(X_new['Age'].median())
X_new = pd.get_dummies(X_new, columns=['Sex'])

print(X_new.isnull().sum())

# {'algorithm': 'ball_tree', 'n_neighbors': 4, 'weights': 'distance'}
model_knnu = KNeighborsClassifier(n_neighbors=4,weights='distance',algorithm='ball_tree')
model_knnu.fit(X_train,y_train)
y_pred = model_knnu.predict(X_new)
print(y_pred)

data_frame = {'PassengerId':df_test['PassengerId'], 'Survived':y_pred}
data_hasil = pd.DataFrame(data=data_frame)
print(data_hasil)

data_hasil.to_csv('out.csv', encoding='utf-8', index=False)