# # -*- coding: utf-8 -*-
#
# import warnings
# warnings.filterwarnings(action='ignore')
#
# import sonya_lib
# import datetime
# from keras.optimizers import Adam
# from keras import callbacks
#
# from scipy import interp
#
#
# # import argparse
# # import locale
# # import os
# # import sys
# import pandas as pd
# import numpy as np
# import warnings
# warnings.filterwarnings(action='ignore')
# import timeit
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# # %matplotlib inline # jupyter 에서만 사용
#
# ## RFE
# from sklearn.feature_selection import RFE, RFECV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline
#
# import keras
# import tensorflow as tf
#
# # from pyimagesearch import datasets
# # from pyimagesearch import models
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# # -------------------------
# # from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
#
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
# from sklearn.metrics import auc, roc_curve  #plot_roc_curve
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
# from keras import losses
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import to_categorical
#
#
# PATH_EXCEL = './BRC2019_CRF_merged_200717 복사본 (copy).xlsx'
# metadata = sonya_lib.get_metadata(PATH_EXCEL)
#
# df_filtered = metadata.dropna(axis=0, how='any').reset_index(drop=True) # NaN drop
# properties = list(df_filtered.columns.values)
# properties.remove('label')
# properties.remove('ID')
# properties.remove('L/R')
# properties.remove('age')
# # properties.remove('hu_diff')
# # properties.remove('aAverage')
# # properties.remove('aSD')
# properties.remove('cAverage')
# # properties.remove('cSD')
# properties.remove('homogeneous')
#
# properties.remove('hetero')
# properties.remove('rim')
# properties.remove('clustered')
# properties.remove('non-mass')
#
# X = df_filtered[properties]
# y = df_filtered['label']
# num_input = len(X.columns)
#
#
# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return optimizer.lr
#     return lr
#
#
# def mlp_model(dropout=0, lr=0.005, l1=9, l2=9):
#     keras.backend.clear_session()
#     ## 모델 구성하기
#     model = Sequential()
#     # print learning rate
#     optimizer = Adam(lr=0.001)
#     lr_metric = get_lr_metric(optimizer)
#
#     model.add(Dense(l1, activation='relu', input_dim=num_input, kernel_initializer='he_normal'))
#     model.add(BatchNormalization())
#     model.add(Dense(l2, activation='relu', kernel_initializer='he_normal'))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout))
#     model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))
#
#     ## 모델 컴파일
#     model.compile(optimizer=Adam(lr), loss=losses.binary_crossentropy, metrics=['accuracy', lr_metric])
#
# #     model.summary()
#     return model
# # def mlp_model(dropout = 0, lr = 0.0001):
# #     keras.backend.clear_session()
# #     ## 모델 구성하기
# #     model = Sequential()
# #
# #     model.add(Dense(32, activation='relu', input_dim=num_input))
# #     model.add(BatchNormalization())
# #     model.add(Dense(32, activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(dropout))
# #     model.add(Dense(1, activation='sigmoid'))
# #
# #     ## 모델 컴파일
# #     model.compile(optimizer=Adam(lr), loss=losses.binary_crossentropy, metrics=['accuracy'])
# #
# # #     model.summary()
# #     return model
#
#
# def cross_validation(model, X, y, nfold=5, nbatch=5, nlr=0.001, l1=16, l2=16):
#     kfold = KFold(n_splits=nfold, shuffle=True)
#     accuracy = []
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     # K-fold cross validation
#     # 학습 데이터를 이용해서 학습
#
#     # ======= tensorboard ========
# #     log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# #     tb_hist = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#
#     i = 1
#     for train_index, validation_index in kfold.split(X, y):
#         kX_train, kX_test = X.iloc[train_index], X.iloc[validation_index]
#         ky_train, ky_test = y.iloc[train_index], y.iloc[validation_index]
#
#         print("======================batch: {}, lr = {}, FOLD: {}====================".format(nbatch, nlr, i))
#         cbks = [callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 ** (epoch // 2)), callbacks.TensorBoard(write_graph=False)]
#         # hist = model.fit(kX_train, ky_train, epochs=500, batch_size=5, validation_data=(kX_test,ky_test),callbacks=[tb_hist])
#         model.fit(kX_train, ky_train, epochs=500, batch_size=nbatch, validation_data=(kX_test, ky_test), callbacks=cbks)
#         # model.save('brc_mlp_model.h5')
#
#         y_val_cat_prob = model.predict_proba(kX_test)
#
#         k_accuracy = '%.4f' % (model.evaluate(kX_test, ky_test)[1])
#         accuracy.append(k_accuracy)
#
#         # roc curve
#         fpr, tpr, t = roc_curve(y.iloc[validation_index], y_val_cat_prob)
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         # final_lr = model.optimizer.lr
#         plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f) ' % (i, roc_auc))
#         i = i + 1
#
#     # 전체 검증 결과 출력
#     print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
#     # print('\nK-fold cross validation mean Accuracy: {}'.format(np.mean(accuracy)))
#
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_auc = auc(mean_fpr, mean_tpr)
#     plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
#     # plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#     # plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC(batch_size: {}, lr: {})'.format(nbatch, nlr))
#     plt.legend(loc="lower right")
#     plt.savefig('./optimization/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '- ROC(batch_size_{}, lr_{}, l1_{}, l2_{}).png'.format(nbatch, nlr,l1,l2))
#     plt.clf()
#     return mean_auc
#
#
#
# def optimize_hyperparameters(_X, _y, lBatch_size, lLearning_rate, layer1_node=16, layer2_node=16):
#     i = 0
#     record = pd.DataFrame(columns=["Batch_size", "Learning_rate", "1st layer", "2nd layer"])
#     # best_auc = 0
#     for nb in lBatch_size:
#         for nlr in lLearning_rate:
#             for l1 in layer1_node:
#                 for l2 in layer2_node:
#                     # df = pd.DataFrame({"Batch_size": [nb], "Learning_rate": [nlr]})
#                     new_record = {'Batch_size': nb, 'Learning_rate': nlr, '1st layer': l1, '2nd layer': l2}
#                     record.loc[i] = new_record
#                     i = i + 1
#                     # my_model = mlp_model(lr=nlr, l1=l1, l2=l2)
#                     # cross_validation(my_model, _X, _y,  nbatch=nb, nlr=nlr, l1=l1, l2=l2)
#
#     # test_loss, test_acc, _ = my_model.evaluate(X_test, y_test)
#     # print('Test acuracy: {}'.format(test_acc))
#
#     record.to_excel('./TEST.xlsx', sheet_name='test', index=False)
#
#
# def rfe(_X, _y):
#     rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
#
#     rfe.fit(_X, _y)
#     for i in range(X.shape[1]):
#         print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
#
#
#
#     # pipeline.get_params()
#     # print("Num Features: {}".format(pipeline.get_params()))
#     # print("Selected Features: %s") % pipeline.support_
#     # print("Feature Ranking: %s") % pipeline.ranking_
#
#
# if __name__ == '__main__':
#     #
#     print("Input param columns\n{}".format(X.columns))
#     print("Input param num: {}".format(num_input))
#     # print(X)
#     # sonya_lib.pause()
#     start_time = timeit.default_timer() # 시작 시간 체크
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#     # ## Grid search
#     # param_grid = {
#     #     'batch_size' : [50, 60, 70, 80, 90],
#     #     'n_nodes' : [5, 10, 15, 20, 25, 30]
#     # }
#     #
#     # estimator = mlp_model()5
#     # grid_search = GridSearchCV(estimator=mlp_model)
#
#
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#     ###
#
#     learning_rate = [0.05]
#     batch_size = [60,80]
#     layer1 = [5,7,9]
#     layer2 = [5,7,9]
#     # learning_rate = [0.005, 0.003, 0.001, 0.0008]
#     # batch_size = [30, 40, 50, 60, 70, 80]
#
#     optimize_hyperparameters(X_train, y_train, batch_size, learning_rate, layer1_node=layer1, layer2_node=layer2)
#
#     terminate_time = timeit.default_timer() # 종료 시간 체크
#
#     total_time = terminate_time - start_time
#
#     # output running time in a nice format.
#     mins, secs = divmod(total_time, 60)
#     hours, mins = divmod(mins, 60)
#
#     print("Total running time: %d:%d:%d.\n" % (hours, mins, secs))


# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore')

import sonya_lib
import datetime
from keras.optimizers import Adam
from keras import callbacks

from scipy import interp

# import argparse
# import locale
# import os
# import sys
import pandas as pd
import numpy as np

import os

warnings.filterwarnings(action='ignore')
import timeit

import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline # jupyter 에서만 사용

## RFE
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import keras
import tensorflow as tf

# from pyimagesearch import datasets
# from pyimagesearch import models

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# -------------------------
# from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import auc, roc_curve  # plot_roc_curve

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

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
    global record_count
    kfold = KFold(n_splits=nfold, shuffle=True)
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
        model.fit(kX_train, ky_train, epochs=500, batch_size=nbatch, validation_data=(kX_test, ky_test), callbacks=cbks, verbose=2)
        # model.save('brc_mlp_model.h5')

        y_val_cat_prob = model.predict_proba(kX_test)

        k_accuracy = '%.4f' % (model.evaluate(kX_test, ky_test)[1])
        accuracy.append(k_accuracy)

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
    print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
    # print('\nK-fold cross validation mean Accuracy: {}'.format(np.mean(accuracy)))
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = str("{}b{}_lr{}_1st{}_2nd{}".format(current_time, nbatch, nlr, l1, l2))
    #
    # ## 모델저장
    # model_json = model.to_json()
    # # if os.path.isfile(".model/{}.json".format(file_name)):
    # with open(".model/{}.json".format(file_name), "x") as json_file:
    #     print("saved model to disk")
    #     json_file.write(model_json)
    # # else:
    #
    #
    # ## weights 저장
    # model.save_weights(".model/{}.h5".format(file_name))
    # print("saved model weights to disk")
    #

    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
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
    # plt.text(0.05, 0.95, 'input: {}\n{}'.format(len(X.columns), X.columns), fontdict=font1)
    # plt.savefig('./optimization/' + datetime.datetime.now().strftime(
    #     "%Y%m%d-%H%M%S") + str(record_count) + '- ROC(input_{}, batch{}, lr_{}, l1_{}, l2_{}).png'.format(len(X.columns), nbatch, nlr, l1, l2))
    # plt.clf()
    plt.show()
    record_count = record_count + 1
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

                    # new_record = {'Batch_size': nb, 'Learning_rate': nlr, '1st layer': l1, '2nd layer': l2,
                    #               "num_input": len(_X.columns), "input_params": _X.columns.values, "mean_ROC": roc_result}
                    # record.loc[i] = new_record


                    ## ------------ TEST_Sensitivity --------------
                    my_model.evaluate(X_origin_test,y_origin_test)
                    predIdxs = my_model.evaluate(X_origin_test,y_origin_test)
                    predIdxs2 = my_model.predict(x=X_origin_test, steps=None)
                    print(predIdxs)
                    print("============================================================")
                    print(predIdxs2)
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
    PATH_EXCEL = './BRC2019_CRF_merged_200717 복사본 (copy).xlsx'
    metadata = sonya_lib.get_metadata(PATH_EXCEL)

    df_filtered = metadata.dropna(axis=0, how='any').reset_index(drop=True)  # NaN drop
    properties = list(df_filtered.columns.values)
    properties.remove('label')
    # properties.remove('cN')
    properties.remove('ID')
    X_origin = df_filtered['cN']
    y_origin = df_filtered['label']
    # num_features = len(X_origin.columns)

    print(df_filtered)

    # batch_size = [20, 40]
    # learning_rate = [0.05, 0.001]
    # layer1 = [7, 9, 11]
    # layer2 = [3, 9, 11]
    batch_size = [20, 30, 40]
    learning_rate = [0.01, 0.05, 0.001]
    layer1 = [3, 5, 7, 9, 11]
    layer2 = [3, 5, 7, 9, 11]

    optimize_hyperparameters(X_origin, y_origin, batch_size, learning_rate, layer1_node=layer1, layer2_node=layer2)


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

