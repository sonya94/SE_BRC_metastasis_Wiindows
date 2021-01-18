# -*- coding: utf-8 -*-
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K



from sklearn.model_selection import train_test_split
# from pyimagesearch import datasets
# from pyimagesearch import models

# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

from scipy import interp

# import numpy as np
# import argparse
# import locale
# import os
# import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


# import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline

# import keras
# import tensorflow as tf

# -------------------------
# from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict

from sklearn.metrics import auc, roc_curve # plot_roc_curve
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import to_categorical


path = './total_836_200622_test.xlsx'
df = pd.read_excel(path)


def homogeneous(row):
    if row['Enhancement'] == 'no-enhancement':
        return 0.25
    elif row['Enhancement'] == 'weak':
        return 0.50
    elif row['Enhancement'] == 'moderate':
        return 0.75
    elif row['Enhancement'] == 'high':
        return 1
    else:
        return 0


def normalize(dataset):
    dataNorm = ((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataset = dataNorm
    return dataset


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

df_filtered = df[[u'시행시나이', 'cT']]
# df_filtered = df[[u'시행시나이','cT','aAverage','aSD','cAverage','cSD']]
# df_filtered['cT'] = normalize(df['cT'])
# df_filtered['cT'] = normalize(df['cT'])
# df_filtered['L/R'] = np.where(df['L/R'] == 'L', 1, 0) # L/R
df_filtered['hu_diff'] = normalize(normalize(df['aAverage']) - normalize(df['cAverage']))
df_filtered['homogeneous'] = df.apply(homogeneous, axis=1)
df_filtered['hetero'] = np.where(df['Enhancement'] == 'hetero', 1, 0)
df_filtered['rim'] = np.where(df['Enhancement'] == 'rim', 1, 0)
df_filtered['clustered'] = np.where(df['Enhancement'] == 'clustered', 1, 0)
df_filtered['non-mass'] = np.where(df['Enhancement'] == 'non-mass', 1, 0)
df_filtered['label'] = df['pN_modify']

df_filtered = df_filtered.rename(columns={u'시행시나이': u'age'})

properties = list(df_filtered.columns.values)
properties.remove('label')
X = df_filtered[properties]
y = df_filtered['label']
num_input = len(X.columns)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# -------------- MLP --------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# # create model
# model = Sequential()
# model.add(Dense(16, activation='relu', input_dim=num_input))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#

# ======= tensorboard ========
# tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# hist = model.fit(X_train, y_train, epochs=1000, batch_size=10, callbacks=[tb_hist])
# ============================

# hist = model.fit(X_train, y_train, epochs=4000, batch_size=5)

# 5. 모델 학습 과정 표시하기

# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# # loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#
# acc_ax.plot(hist.history['acc'], 'b', label='train acc')
# # acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
#
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
#
# plt.show()


# test_loss, test_acc = model.evaluate(X_test, y_test)

# print('Test acuracy: ', test_acc)

# ============================================================================


kfold = KFold(n_splits=5, shuffle=True)

accuracy = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


# #======== no k-fold ========
#
# def plot_roc_curve(fpr, tpr):
#     plt.plot(fpr, tpr)
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.show()
#
#
# for train, validation in kfold.split(X_train, y_train):
#
#     kX_train, kX_test = X_train.iloc[train], X_train.iloc[validation]
#     ky_train, ky_test = y_train.iloc[train], y_train.iloc[validation]
#
#     model = Sequential()
#
#     model.add(Dense(16, activation='relu', input_dim=num_input))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     # 학습 데이터를 이용해서 학습
#     hist = model.fit(kX_train, ky_train, epochs=1000, batch_size=5)
#
#     y_val_cat_prob = model.predict_proba(X_test)
#     fpr, tpr, thresholds = roc_curve(y_test, y_val_cat_prob)
#
#
#
#     # roc curve
#     # fpr, tpr, thresholds = roc_curve(y_train.iloc[validation] ,hist[:, 1])
#     # tprs.append(interp(mean_fpr, fpr, tpr))
#     # roc_auc = auc(fpr, tpr)
#     # aucs.append(roc_auc)
#     # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     # i = i + 1
#     # 테스트 데이터를 이용해서 검증
#     k_accuracy = '%.4f' % (model.evaluate(kX_test, ky_test)[1])
#     accuracy.append(k_accuracy)
#
# # 전체 검증 결과 출력
# print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
# print('\nK-fold cross validation mean Accuracy: {}'.format(np.mean(accuracy)))
model = Sequential()


model.add(Dense(32, activation='relu', input_dim=num_input))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0))
model.add(Dense(1, activation='sigmoid'))

 # print learning rate
optimizer = Adam(lr=0.001)
lr_metric = get_lr_metric(optimizer)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', lr_metric])
# model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    print("saved model to disk")
    json_file.write(model_json)

model.summary()


#======== k-fold ========


i = 1
for train_index, validation_index in kfold.split(X_train, y_train):
    kX_train, kX_test = X_train.iloc[train_index], X_train.iloc[validation_index]
    ky_train, ky_test = y_train.iloc[train_index], y_train.iloc[validation_index]
    print("===================================FOLD: {}===================================".format(i))

    cbks = [callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 **(epoch//2)), callbacks.TensorBoard(write_graph=False)]
    # 학습 데이터를 이용해서 학습
    model.fit(kX_train, ky_train, epochs=5, batch_size=5, callbacks=cbks, verbose=2)
    model.save('brc_mlp_model.h5')
    y_val_cat_prob = model.predict_proba(kX_test)

    k_accuracy = '%.4f' % (model.evaluate(kX_test, ky_test)[1])
    accuracy.append(k_accuracy)

    # roc curve
    fpr, tpr, t = roc_curve(y[validation_index], y_val_cat_prob)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # final_lr = model.optimizer.lr
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f) ' % (i, roc_auc))
    i = i + 1

# 전체 검증 결과 출력
print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
# print('\nK-fold cross validation mean Accuracy: {}'.format(np.mean(accuracy)))



plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % mean_auc, lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
# plt.text(0.32,0.7,'More accurate area',fontsize = 12)
# plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

test_loss, test_acc, _ = model.evaluate(X_test, y_test)
print('Test acuracy: {}'.format(test_acc))