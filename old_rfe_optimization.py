# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings(action='ignore')
import os, sys
import pandas as pd
import numpy as np

def importOwnLib():
    if '/home/miruware/aProjects/lib' not in sys.path:
        sys.path.append('/home/miruware/aProjects/lib')
        print("lib path is successfully appended.")
    else:
        print("lib path is already exists.")

importOwnLib()
import sonyalib as sonya

import importlib
importlib.reload(sonya)

import datetime
import timeit

from scipy import interp

import keras
from keras.optimizers import Adam
from keras import callbacks, losses
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
# import argparse
# import locale

import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline # jupyter 에서만 사용

## RFE
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, roc_curve
from sklearn.pipeline import Pipeline

from sklearn.metrics import auc, roc_curve  # plot_roc_curve

# -------------------------
# from sklearn.preprocessing import LabelBinarizer, MinMaxScaler



record_count = 0


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def mlp_model(num_input, dropout=0, lr=0.005, l1=9, l2=9):
    keras.backend.clear_session()
    ## 모델 구성하기
    model = Sequential()
    # print learning rate
    optimizer = Adam(lr=0.001)
    lr_metric = get_lr_metric(optimizer)

    model.add(Dense(l1, activation='relu', input_dim=num_input, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(l2, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))

    ## 모델 컴파일
    model.compile(optimizer=Adam(lr), loss=losses.binary_crossentropy, metrics=['accuracy', lr_metric])

    # model.summary()
    return model


def cross_validation(model, X, y, nfold=5, nbatch=5, nlr=0.001, l1=16, l2=16):
    # global record_count
    kfold = StratifiedKFold(n_splits=nfold, shuffle=True)
    accuracy = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # K-fold cross validation
    # 학습 데이터를 이용해서 학습

    i = 1
    for train_index, validation_index in kfold.split(X, y):

        kX_train, kX_test = X.iloc[train_index], X.iloc[validation_index]
        ky_train, ky_test = y.iloc[train_index], y.iloc[validation_index]

        print("======================batch: {}, lr = {}, FOLD: {}====================".format(nbatch, nlr, i))
        cbks = [callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 ** (epoch // 2)),
                callbacks.TensorBoard(write_graph=False)]
        # hist = model.fit(kX_train, ky_train, epochs=500, batch_size=5, validation_data=(kX_test,ky_test),callbacks=[tb_hist])
        model.fit(kX_train, ky_train, epochs=500, batch_size=nbatch, validation_data=(kX_test, ky_test), callbacks=cbks,
                  verbose=2)
        y_val_cat_prob = model.predict_proba(kX_test)

        k_accuracy = '%.4f' % (model.evaluate(kX_test, ky_test)[1])
        accuracy.append(k_accuracy)

        # model_name = target_dir + '/' + str(i) + '_AUC' + str(int(float(k_accuracy) * 100))
        # model_json = model.to_json()
        #
        # with open('{}.json'.format(model_name), 'w') as json_file:
        #     json_file.write(model_json)  # save model per fold
        #
        # model.save_weights('{}.h5'.format(model_name))  # save weight per fold

        # roc curve
        fpr, tpr, t = roc_curve(y.iloc[validation_index], y_val_cat_prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # final_lr = model.optimizer.lr
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f) ' % (i, roc_auc))
        i = i + 1

        ## -------- Sensitivity -------


        ## ----------------------------

    # 전체 검증 결과 출력

    test_loss, test_acc, _ = my_model.evaluate(X_test, y_test)
    test_acc_str = 'Test acuracy: {}'.format(test_acc)
    # print('Test acuracy: {}'.format(test_acc))
    print(test_acc_str)
    print('\nK-fold cross validation Accuracy: {}'.format(accuracy))


    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    # plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    # plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    font1 = {'family': 'serif',
             'color': 'darkred',
             'weight': 'normal',
             'size': 10}
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC(batch: {}, lr: {}, l1: {}, l2: {})'.format(nbatch, nlr, l1, l2))
    plt.legend(loc="lower right")
    plt.text(0.05, 0.95, test_acc_str, fontdict=font1)
    fig_dir = './model_old_hyper_tuning/base_model_test00_graph/'
    sonya.createFolder(fig_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(fig_dir + 'AUC(%0.2f)' % (mean_auc) + 'Acc(%0.2f)' % (test_acc) + current_time + ' - ROC(input_{}, batch{}, lr_{}, l1_{}, l2_{}).png'.format(len(X.columns), nbatch, nlr, l1, l2))

    # plt.show()
    plt.clf()  # clear the current figure
    return mean_auc


def optimize_hyperparameters(_X, _y, lBatch_size, lLearning_rate, layer1_node=16, layer2_node=16):
    X_origin_train, X_origin_test, y_origin_train, y_origin_test = train_test_split(_X, _y, test_size=0.3, random_state=0)

    print(X_origin_train.shape, X_origin_test.shape, y_origin_train.shape, y_origin_test.shape)

    i = 0
    record = pd.DataFrame(columns=["Batch_size", "Learning_rate", "1st layer", "2nd layer", "num_input", "input_params", "mean_ROC"])
    # best_auc = 0
    for nb in lBatch_size:
        BS = nb
        for nlr in lLearning_rate:
            for l1 in layer1_node:
                for l2 in layer2_node:
                    # df = pd.DataFrame({"Batch_size": [nb], "Learning_rate": [nlr]})

                    # my_model = mlp_model(len(_X.columns), lr=nlr, l1=l1, l2=l2)
                    my_model = mlp_model(1, lr=nlr, l1=l1, l2=l2)
                    roc_result = cross_validation(my_model, X_origin_train, y_origin_train,  nbatch=nb, nlr=nlr, l1=l1, l2=l2)

                    print(roc_result)
                    # new_record = {'Batch_size': nb, 'Learning_rate': nlr, '1st layer': l1, '2nd layer': l2,
                    #               "num_input": len(_X.columns), "input_params": _X.columns.values, "mean_ROC": roc_result}
                    # record.loc[i] = new_record


                    ## ------------ TEST_Sensitivity --------------
                    my_model.evaluate(X_origin_test,y_origin_test)
                    predIdxs = my_model.evaluate(X_origin_test,y_origin_test)
                    predIdxs2 = my_model.predict(x=X_origin_test, steps=None)
                    # print(predIdxs)
                    # print("============================================================")
                    # print(predIdxs2)
                    # predIdxs = my_model.predict(x=X_origin_test, steps=(len(_X) // BS) + 1)
                    # predIdxs = my_model.predict(x=X_origin_test, steps=(totalTest // BS) + 1)

                    # predIdxs = np.argmax(predIdxs, axis=1)
                    # cm = confusion_matrix(y_origin_test, predIdxs)
                    # cm2 = confusion_matrix(y_origin_test, predIdxs).ravel()
                    # print(cm)
                    # print(cm2)
                    # # total = sum(sum(int(cm)))
                    # # acc = (cm[0, 0] + cm[1, 1]) / total
                    # sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                    # specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                    #
                    # print(cm)
                    # print(cm2)
                    # # print("acc: {:.4f}".format(acc))
                    # print("sensitivity: {:.4f}".format(sensitivity))
                    # print("specificity: {:.4f}".format(specificity))


                    ## --------------------------------------------

                    i = i + 1

    # test_loss, test_acc, _ = my_model.evaluate(X_test, y_test)
    # print('Test acuracy: {}'.format(test_acc))

    # record.to_excel('./records/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(len(_X.columns)) + '.xlsx', sheet_name='record', index=False)
    # return record


def rfe(_X, _y, nFeatures=3):

    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=nFeatures)
    rfe.fit(_X, _y)
    # print(_X.columns[rfe.get_support(indices=True)].tolist())

    # for i in range(_X.shape[1]):
    #     print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

    # print("Selected Features: %s") % pipeline.support_
    # print("Feature Ranking: %s") % pipeline.ranking_

    ## return the list of selected features
    return _X.columns[rfe.get_support(indices=True)].tolist()


if __name__ == '__main__':

    print("This is local file")
    # test_model = mlp_model(5)
    # print(os.getcwd())
    # model_json = test_model.to_json()
    # filename = str("test2")
    # path = ".model/{}.json".format(filename)
    # with open(path, mode='x') as test_file:
    #     print("saved model to disk")
    #     test_file.write(model_json)

    # sonya_lib.pause()

    start_time = timeit.default_timer()  # 시작 시간 체크
    # ==== Step 1. Load original dataset
    file_path0 = './BRC_input_201116_train.xlsx'
    file_path1 = './BRC_input_201116_test.xlsx'
    meta_train = sonya.get_normalized_metadata(file_path0)
    meta_test = sonya.get_normalized_metadata(file_path1)

    properties = [  # total 18
        'sex',
        'age',
        'LR',
        'cT',
        'cN',
        'cAverage',
        'cSD',
        'aAverage',
        'aSD',
        'lMax',
        'homogeneous',
        'hetero',
        'rim',
        'clustered',
        'non-mass',
        'AorCa',
        'LymAo',
        'LymCa'
    ]
    num_properties = len(properties)

    target_dir = './model_old_hyper_tuning/base_model_test00'
    sonya.createFolder(target_dir)
    X_train = meta_train[properties]
    y_train = meta_train['label']

    X_test = meta_test[properties]
    y_test = meta_test['label']
    # batch_size = [20, 40]
    # learning_rate = [0.05, 0.001]
    # layer1 = [7, 9, 11]
    # layer2 = [3, 9, 11]
    batch_size = [5, 10, 15, 20, 25, 30, 35, 40]
    learning_rate = [0.1, 0.05, 0.01, 0.005]
    layer1 = [3, 5, 7, 9, 11]
    layer2 = [3, 5, 7, 9, 11]

    # optimize_hyperparameters(X_train, y_train, batch_size, learning_rate, layer1_node=layer1, layer2_node=layer2)

    ## ========== >>>>>>>>>> optimize_hyperparameters <<<<<<<<<< ==========

    # X_origin_train, X_origin_test, y_origin_train, y_origin_test = train_test_split(_X, _y, test_size=0.3,random_state=0)
    #
    # print(X_origin_train.shape, X_origin_test.shape, y_origin_train.shape, y_origin_test.shape)

    i = 0
    record = pd.DataFrame(
        columns=["Batch_size", "Learning_rate", "1st layer", "2nd layer", "num_input", "input_params", "mean_ROC"])
    # best_auc = 0
    for nb in batch_size:
        BS = nb
        for nlr in learning_rate:
            for l1 in layer1:
                for l2 in layer2:
                    my_model = mlp_model(num_properties, lr=nlr, l1=l1, l2=l2)
                    roc_result = cross_validation(my_model, X_train, y_train, nbatch=nb, nlr=nlr, l1=l1,l2=l2)

                    print(roc_result)
                    # sonya.pause()
                    # new_record = {'Batch_size': nb, 'Learning_rate': nlr, '1st layer': l1, '2nd layer': l2,
                    #               "num_input": len(_X.columns), "input_params": _X.columns.values, "mean_ROC": roc_result}
                    # record.loc[i] = new_record

                    ## ------------ TEST_Sensitivity --------------
                    # predIdxs = my_model.evaluate(X_test, y_test)
                    # predIdxs2 = my_model.predict(x=X_test, steps=None)
                    # print(predIdxs)
                    # print("============================================================")
                    # print(predIdxs2)
                    # predIdxs = my_model.predict(x=X_origin_test, steps=(len(_X) // BS) + 1)
                    # predIdxs = my_model.predict(x=X_origin_test, steps=(totalTest // BS) + 1)

                    # predIdxs = np.argmax(predIdxs, axis=1)
                    # cm = confusion_matrix(y_origin_test, predIdxs)
                    # cm2 = confusion_matrix(y_origin_test, predIdxs).ravel()
                    # print(cm)
                    # print(cm2)
                    # # total = sum(sum(int(cm)))
                    # # acc = (cm[0, 0] + cm[1, 1]) / total
                    # sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                    # specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                    #
                    # print(cm)
                    # print(cm2)
                    # # print("acc: {:.4f}".format(acc))
                    # print("sensitivity: {:.4f}".format(sensitivity))
                    # print("specificity: {:.4f}".format(specificity))

                    ## --------------------------------------------

                    i = i + 1

                    # test_loss, test_acc, _ = my_model.evaluate(X_test, y_test)
                    # print('Test acuracy: {}'.format(test_acc))

    # record.to_excel('./records/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(len(_X.columns)) + '.xlsx', sheet_name='record', index=False)
    # return record

    # roc_result = cross_validation(my_model, X_origin_train, y_origin_train, nbatch=20, nlr=0.05, l1=9, l2=3)

    # print(X_origin.columns[8])
    # ==== Step 2. RFE
    # for num_features in range(1, 4):
    #     list_selected = rfe(X_origin_train, y_origin_train, num_features) ## it returns the list of selected features
    #
    #     # ==== Step 3. get the new dataset of selected features
    #     X_selected_train = X_origin_train[list_selected]
    #
    #     # ==== Step 4. Optimize the hyper parameters of selected inputs
    #     batch_size = [20, 40]
    #     learning_rate = [0.05, 0.001]
    #     layer1 = [7, 9, 11]
    #     layer2 = [3, 9, 11]
    #     # 8월 7일 hyper parameters
    #     # batch_size = [20, 40, 60, 80, 100]
    #     # learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
    #     # layer1 = [3, 5, 7, 9, 11]
    #     # layer2 = [3, 5, 7, 9, 11]
    #     optimize_hyperparameters(X_selected_train, y_origin_train, batch_size, learning_rate, layer1_node=layer1, layer2_node=layer2)



    terminate_time = timeit.default_timer()  # 종료 시간 체크

    total_time = terminate_time - start_time

    # output running time in a nice format.
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    print("Total running time: %d:%d:%d.\n" % (hours, mins, secs))

    # # sonya_lib.pause()
    # start_time = timeit.default_timer()  # 시작 시간 체크
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #
    #
    # learning_rate = [0.05]
    # batch_size = [60, 80]
    # layer1 = [5, 7, 9]
    # layer2 = [5, 7, 9]
    #
    # optimize_hyperparameters(X_train, y_train, batch_size, learning_rate, layer1_node=layer1, layer2_node=layer2)
    #
    # terminate_time = timeit.default_timer()  # 종료 시간 체크
    #
    # total_time = terminate_time - start_time
    #
    # # output running time in a nice format.
    # mins, secs = divmod(total_time, 60)
    # hours, mins = divmod(mins, 60)

