# 'import library'
# ==================================================
import sys, os

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

# ==================================================
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.optimizers import Adam


def create_model(num_input=1, dropout=0, learning_rate=0.005, neurons1=9, neurons2=9):
    keras.backend.clear_session()

    ## 모델 구성하기
    model = Sequential()

    model.add(Dense(neurons1, activation='relu', input_dim=num_input, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(neurons2, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))

    ## 모델 컴파일
    model.compile(optimizer=Adam(learning_rate), loss=losses.binary_crossentropy, metrics=['accuracy'])

    # model.summary()

    return model


if __name__ == '__main__':

    # fit random seed for reproductivity
    seed = 7
    np.random.seed(seed)

    # Load metadata

    file_path0 = './BRC_input_201116.xlsx'
    metadata = sonya.get_original_metadata(file_path0)

    # properties = list(metadata.columns.values)
    # properties.remove('label')
    # X = metadata[properties]
    # y = metadata['label']

    X = metadata[[
        # 'sex',
        # 'age',
        # 'LR',
        # 'cT',
        # 'cN',
        # 'cAverage',
        # 'cSD',
        # 'aAverage',
        # 'aSD',
        # 'lMax',
        # 'homogeneous',
        # 'hetero',
        # 'rim',
        # 'clustered',
        # 'non-mass',
        # 'AorCa',
        'LymAo',
        # 'LymCa'
    ]]

    y = metadata['label']

    num_input = len(X.columns)

    # seperate dataset to train data & test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    # create model
    # model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=batch_size)
    model = KerasClassifier(build_fn=create_model)

    batch = [10, 20, 30, 40]
    epochs = [10, 50, 100]
    lr = [0.001, 0.005, 0.01, 0.05]
    neurons1 = [3, 5, 7, 9, 11]
    neurons2 = [3, 5, 7, 9, 11]

    grid = dict(
        neurons1=neurons1,
        neurons2=neurons2,
        batch_size=batch,
        learning_rate=lr
    )

    grid = GridSearchCV(estimator=model, param_grid=grid, cv=10)
    grid_result = grid.fit(X_train, y_train) # 최적의 hyperparameters를 찾는데 testset을 사용하지 않았다.

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
