import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.optimizers import Adam

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


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def create_model(num_input=18, dropout=0, learning_rate=0.005, neurons1=9, neurons2=9):
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

# fit random seed for reproductivity
seed = 7
np.random.seed(seed)

# load dataset
PATH_EXCEL = './BRC_input_201116.xlsx'
metadata = sonya.get_metadata(PATH_EXCEL)

df_filtered = metadata.dropna(axis=0, how='any').reset_index(drop=True)  # NaN drop
properties = list(df_filtered.columns.values)
properties.remove('label')
# properties.remove('cN')
properties.remove('ID')
X_origin = df_filtered[properties]
y_origin = df_filtered['label']
num_input = len(properties)
print(num_input)

X_train, X_test, y_train, y_test = train_test_split(X_origin, y_origin, test_size=0.3, random_state=0)

x, y = X_train, y_train

# create model
# model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=batch_size)
model = KerasClassifier(build_fn=create_model, verbos=0)


batch_size = [10, 20, 30, 40]
epochs = [10, 50, 100]
learning_rate = [0.01, 0.05, 0.001]
neurons1 = [3, 5, 7, 9, 11]
neurons2 = [3, 5, 7, 9, 11]

param_grid = dict(
    # learning_rate=learning_rate,
    neurons1=neurons1,
    neurons2=neurons2,
    batch_size=batch_size,
    learning_rate=learning_rate
)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
grid_result = grid.fit(x, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
