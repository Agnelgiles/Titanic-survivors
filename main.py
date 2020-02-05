import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from transformData import create_data_frame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

log_reg_params = {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
pd.set_option('display.max_columns', None)
orginalTrainData = pd.read_csv("data/train.csv")
orginalTestX = pd.read_csv("data/test.csv")
orginalTestY = pd.read_csv("data/gender_submission.csv")
trainY = orginalTrainData['Survived']
trainX = create_data_frame(orginalTrainData.drop(['Survived'], axis=1))
testX = create_data_frame(orginalTestX)
testY = orginalTestY.drop(['PassengerId'], axis=1)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(trainX, trainY)
log_reg = grid_log_reg.best_estimator_
log_reg.fit(trainX, trainY)
y_pred_log_reg = log_reg.predict(testX)
log_reg_cf = confusion_matrix(testY, y_pred_log_reg)

print(classification_report(testY, y_pred_log_reg))
print("Logistic regression accuracy is {}%".format(np.round((accuracy_score(testY, y_pred_log_reg) * 100), 2)))
fig, ax = plt.subplots(1, 1)
sns.heatmap(log_reg_cf, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['', ''], fontsize=14, rotation=90)
ax.set_yticklabels(['', ''], fontsize=14, rotation=360)
plt.show()
