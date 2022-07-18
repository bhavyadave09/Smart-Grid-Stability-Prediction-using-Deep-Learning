```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Nadam

from datetime import datetime
```


```python
def plotf(f_data, f_y_feature, f_x_feature, f_index=-1):

    for f_row in f_data:
        if f_index >= 0:
            f_color = np.where(f_data[f_row].index == f_index,'red','silver')
            f_hue = None
        else:
            f_color = np.where(f_data[f_row].index == f_index,'green','darkseagreen')
            f_hue = None
    
    f_fig, f_a = plt.subplots(1, 2, figsize=(16,4))
    
    f_chart1 = sns.histplot(f_data[f_x_feature], ax=f_a[0], kde=False, color='silver')
    f_chart1.set_xlabel(f_x_feature,fontsize=10)
    
    if f_index >= 0:
        f_chart2 = plt.scatter(f_data[f_x_feature], f_data[f_y_feature], c=f_color, edgecolors='w')
        f_chart2 = plt.xlabel(f_x_feature, fontsize=10)
        f_chart2 = plt.ylabel(f_y_feature, fontsize=10)
    else:
        f_chart2 = sns.scatterplot(x=f_x_feature, y=f_y_feature, data=f_data,c=f_color, hue=f_hue, legend=False)
        f_chart2.set_xlabel(f_x_feature,fontsize=10)
        f_chart2.set_ylabel(f_y_feature,fontsize=10)

    plt.show() 
```


```python
def correlation(f_data, f_feature, f_number):

    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(f_correlation, mask=f_mask, vmin=-1, vmax=1, square=True,
                    center=0, annot=True, annot_kws={"size": 8}, cmap="Greens")
    plt.show()
```


```python
sns.set()
start_time = datetime.now()

data = pd.read_csv('data/SmartGridStabilityAugmented.csv')

map1 = {'unstable': 0, 'stable': 1}
data['stabf'] = data['stabf'].replace(map1)

data = data.sample(frac=1)
```


```python
data.head()
```


```python
for column in data.columns:
    plotf(data, 'stab', column, -1)
```


```python
data.p1.skew()
```


```python
print(f'In the original dataset, the split of "unstable" (0) and "stable" (1) observations:')
print(data['stabf'].value_counts(normalize=True))
```


```python
correlation(data, 'stabf', 14)
```


```python
X = data.iloc[:, :12]
y = data.iloc[:, 13]

X_train1 = X.iloc[:54000, :]
y_train1 = y.iloc[:54000]

X_test1 = X.iloc[54000:, :]
y_test1 = y.iloc[54000:]

ratio_training = y_training.value_counts(normalize=True)
ratio_testing = y_testing.value_counts(normalize=True)
ratio_training, ratio_testing
```


```python
X_train1 = X_train1.values
y_train1 = y_train1.values

X_test1 = X_test1.values
y_test1 = y_test1.values
```


```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
# ANN initialization
classifier = Sequential()

# Input layer and first hidden layer
classifier.add(Dense(units = 288 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

#Second hidden layer
classifier.add(Dense(units = 288, kernel_initializer = 'uniform', activation = 'relu'))

# Third hidden layer
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))

# Fourth hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Single-node output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# ANN compilation
classifier.compile(optimizer = "nadam", loss = 'binary_crossentropy', metrics = ['accuracy'])

```


```python
cross_val_round = 1

for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_training):
    x_train, x_val = X_train1[train_index], X_train1[val_index]
    y_train ,y_val = y_train1[train_index], y_train1[val_index]
    classifier.fit(x_train, y_train, epochs=50)
    print(f'\nModel evaluation - Round {cross_val_round}: {classifier.evaluate(x_val, y_val)}\n')
    cross_val_round += 1
```


```python
print(X_test1.shape)
#X_testing=X_test1.reshape(6000,12,1)
y_pred = classifier.predict(X_test1)
#y_pred = y_pred.reshape(6000,1)
```


```python
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
print(y_pred)
```


```python
ConfusionMatrix = confusion_matrix(y_test1, y_pred)
print(ConfusionMatrix)

```


```python
print(f'Accuracy: {(ConfusionMatrix[0][0] + ConfusionMatrix[1][1]) / ConfusionMatrix.sum()*100:.2f}%')
```
