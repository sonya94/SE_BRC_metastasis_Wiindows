print(__doc__)
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
## ===================================================
# Author : Soyoung Park
# Creation Time : 27/11/2020 12:13 PM
# Description: Recursive Feature Elimination code
# Steps: 1. 0_model_tuning / 2. 0_model_training / 3. 2_rfe

## ====================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.visible_device_list = "3, 4"
# set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
import os

# import essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import model_from_json


# import own library
import os, sys
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


# transform CSV to DataFrame
file_path0 = './BRC_input_201116.xlsx'
metadata = sonya.get_original_metadata(file_path0)





# seperate dataset to train data & test data
# choose variable


# X = metadata[properties]


# for Regression -------------------------------------------------------------

X = metadata[[
    # 'sex',
    # 'age',
    # 'LR',
    #'cT',
    # 'cN',
    # 'cAverage',
    # 'cSD',
    'aAverage',
    # 'aSD',
    'lMax',
    # 'homogeneous',
    # 'hetero',
    # 'rim',
    # 'clustered',
    # 'non-mass',
    'AorCa',
    'LymAo',
    'LymCa'
]]
y = metadata['label']
# check data type
print(X.info())
print('\n')
# view summary of data statistics
print(X.describe())

num_input = len(X.columns)
print(num_input)
# y = df1[['V00GT']]  # dependent variable : target vector
# ---------------------------------------------------------------------------

'--------------------- load model --------------------------- '
# ==================================================================
target_dir = './base_model_RFE_test00_PR{}'.format(num_input)  # 0-th elimination
# target_dir = './result_RFE_test01_AUC62'     # 1-th elimination        # rfe_1 Skin Flap vs GI flap_0
# ==================================================================



raw_data = X
data_columns_size = raw_data.columns.size

raw_data_reshape = raw_data
data_trans = raw_data_reshape

x_raw_data = data_trans

from scipy.stats import zscore

X_total = x_raw_data

x_input = X_total
numvars = x_input.shape[1]

x_sens_base2 = x_input.median()  # Column별 median
x_sens_base = x_input.mean()  # Column별 median
for seq_id in range(numvars):
    if len(x_input.iloc[:, seq_id].unique()) == 2:
        x_sens_base[seq_id] = 0  # Binary var는 0으로

Sens_test_num = 10
Pred_rx1 = np.zeros((Sens_test_num, len(x_sens_base)))  # 변수별 Sens_test결과 array
# Pred_rx1 = pd.DataFrame()
Pred_diff_stage = []

min(x_input.iloc[:, 1])
max(x_input.iloc[:, 1])

x_input = np.array(x_input)

# -------------------------------------------------------
import os

def list_files_subdir(destpath, ext):
    filelist = []
    for path, subdirs, files in os.walk(destpath):
        for filename in files:
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                if filename.endswith(ext):
                    filelist.append(f)
    filelist.sort()
    return filelist


# filelist = list_files_subdir(strdirectory_fc, 'txt')
h5_list = list_files_subdir(target_dir, 'h5')
json_list = list_files_subdir(target_dir, 'json')
# -------------------------------------------------------

index_list = []
value_list = []

df_effect = pd.DataFrame(raw_data.columns)
for i in range(len(h5_list)):
    index_list = []
    value_list = []
    print(h5_list[i])
    print(json_list[i])
    # load model & weights
    json_name = (json_list[i])
    # json_file = open(fileName+".json", "r")
    json_file = open(json_name, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #
    h5_file = h5_list[i]
    loaded_model.load_weights(h5_file)
    print("Loaded model from disk", h5_file, ".json/h5")
    print(json_name, h5_file)
    '--------------------- end of load model --------------------------- '
    #
    #
    # sigma = -1
    sigma = 2
    for seq_id in range(numvars):
        # for seq_id in range(1):
        X_sens_test = np.tile(x_sens_base, (Sens_test_num, 1))  # Make test base
        if (len(np.unique(x_input[:, seq_id])) == 2):
            X_sens_test[(Sens_test_num // 2):, seq_id] = 1
        else:
            if sigma == -1:
                X_sens_test[:, seq_id] = np.linspace(min(x_input[:, seq_id]),
                                                     max(x_input[:, seq_id]), Sens_test_num)
            elif sigma > 0:
                x_avg = x_input[:, seq_id].mean();
                x_sd = x_input[:, seq_id].std()
                X_sens_test[:, seq_id] = np.linspace(x_avg - (sigma * x_sd), x_avg + (sigma * x_sd), Sens_test_num)
        loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
        y_pred = loaded_model.predict_proba(X_sens_test)
        # Pred_rx1.iloc[:, seq_id] = y_pred     # class를 기준으로 함
        Pred_rx1[:, seq_id] = y_pred[:, 0]  # '0' for sigmoid, '1' for softmax # class를 기준으로 함
        # y_pred_softmax = resto_sess.run(tf.nn.softmax(y_pred))    # Pred_rx0[:, seq_id] = y_pred_softmax[:, 0]
        Pred_diff = np.max(Pred_rx1[:, seq_id], axis=0) - np.min(Pred_rx1[:, seq_id], axis=0)
        Pred_diff_stage.append(Pred_diff)
        print(Pred_diff)
    print("------------------------")
    df_effect[1 + i] = Pred_diff_stage
    Pred_diff_stage = []

df_effect[1 + numvars] = df_effect.mean(axis=1)
df_effect[2 + numvars] = df_effect.std(axis=1)
df_effect = df_effect.rename(columns={1 + numvars: 'mean'})
df_effect = df_effect.rename(columns={2 + numvars: 'std'})
df_effect1 = df_effect.sort_values(by=['mean'], axis=0)
df_effect1['fNum'] = df_effect1.index
df_effect1 = df_effect1.reset_index(drop=True)
df_effect0 = df_effect1
df_effect1 = df_effect1.filter([0, 'fNum', 'mean', 'std'])
df_effect2 = df_effect1.filter([0, 'fNum', 'mean'])

df_effect0.to_csv(target_dir + "/rfe" + '.txt')

elim_idx, elim_pred_diff = min(enumerate(df_effect['mean']), key=lambda x: x[1])

elim_fName = X_total.columns[elim_idx]
print("Elim target : ", "#", elim_idx, " ", elim_fName, "w/ effect of ", elim_pred_diff)
print(df_effect2)

plt.plot(Pred_rx1[:, elim_idx], label='Sensitivity'.format(elim_pred_diff, elim_fName))
plt.title('Sensitivity of {}'.format(elim_fName))
plt.ylabel("Probability Effect")
plt.xlabel("Test Stage")
plt.legend(loc="best")
plt.savefig('./plots/Sensitivity of {}.png'.format(elim_fName))
plt.show()
