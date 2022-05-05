# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 14:14
# @Author  : 周智勇
# @FileName:C的变化对逻辑回归的影响.py
# @Software:PyCharm


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

bc_data = datasets.load_breast_cancer()
X = bc_data.data
Y = bc_data.target
l1_train = []
l2_train = []
l1_test = []
l2_test = []

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.3)  # , random_state=50

for i in np.linspace(0.05, 2, 19):
    LR_L1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)
    LR_L2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)

    LR_L1 = LR_L1.fit(X_train, Y_train)
    l1_train.append(accuracy_score(LR_L1.predict(X_train), Y_train))
    l1_test.append(accuracy_score(LR_L1.predict(X_test), Y_test))

    LR_L2 = LR_L2.fit(X_train, Y_train)
    l2_train.append(accuracy_score(LR_L2.predict(X_train), Y_train))
    l2_test.append(accuracy_score(LR_L2.predict(X_test), Y_test))

print(LR_L1.n_iter_)

graph = [l1_train, l2_train, l1_test, l2_test]
color = ['green', 'red', 'black', 'purple']
label = ['l1_train', 'l2_train', 'l1_test', 'l2_test']

plt.figure(figsize=(10, 8), dpi=80)
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1, 19), graph[i], color=color[i], label=label[i])

plt.xlabel('the value of the C')
plt.ylabel('accuracy_score')
plt.legend(loc='best')
plt.show()
