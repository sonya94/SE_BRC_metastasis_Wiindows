print(__doc__)
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.visible_device_list = "3, 4" ## 사용하고자 하는 gpu 설정
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
import os

# import essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sonya_lib

# transform CSV to DataFrame
# file_path0 = './colon-surgery2013_2014_2015_2016_2017_2018_2019pt_SSI-unlock_reco_20200311semifinal_merge.csv'
# file_path1 = './colon-surgery2013_2014_2015_2016_2017pt_SSI-unlock_reco_20200311semifinal_merge.csv'
#
# file_path01 = '/data/SSI/colon-surgery2018pt_SSI-unlock_reco_20200311semifinal.txt'
# file_path02 = '/data/SSI/colon-surgery2019pt_SSI-unlock_reco_20200311semifinal.txt'
#
# df = pd.read_csv(file_path0, sep=',', encoding='utf-8', header=0, index_col=0)
# df1 = pd.read_csv(file_path1, sep=',', encoding='utf-8', header=0, index_col=0)  # train 2013~17
# # df2 = pd.read_csv(file_path2, sep=',', encoding='utf-8', header=0, index_col=0)    # test 2018~19
#
# df01 = pd.read_csv(file_path01, sep=',', encoding='utf-8', header=0, index_col=0)
# df02 = pd.read_csv(file_path02, sep=',', encoding='utf-8', header=0, index_col=0)
# # df = pd.concat([df01, df02, df03])
# df2 = pd.concat([df01, df02])
#
# # check data type
# print(df.info())
# print('\n')

#========= 적용중=====
PATH_EXCEL = './BRC2019_CRF_merged_200717 복사본 (copy).xlsx'


df = sonya_lib.get_metadata_no_normalize(PATH_EXCEL)

# view summary of data statistics
print(df.describe())

df_filtered = df.dropna(axis = 0, how = 'any').reset_index(drop=True) # NaN drop
properties = list(df_filtered.columns.values)
properties.remove('label')
properties.remove('ID')
X = df_filtered[properties]
y = df_filtered['label']


# seperate dataset to train data & test data
# choose variable
' make target feature matrix & target '

# for Regression -------------------------------------------------------------
print('df.columns: ', df.columns)

# X = df1[[  # 13~17
#     'V01LOSaftOP', 'V02EmergencyOperation', 'V03RiskIndex', 'V04DiseaseCode',
#     'V05DateReAdm', 'V06DateReEmergVisit', 'V07DdateReOp', 'V08AbdominalScan',
#     'V09PCD', 'V10PCD_abdomen', 'V11PCD_liver', 'V12PCD_lung',
#     'V13M41', 'V14M42', 'V15M44', 'V16M45', 'V17M47',
#     'V18M50', 'V19M52', 'V20M61', 'V21GramStain',
#     'V22BloodCultureN', 'V23BloodCulture', 'V24BodyTempN', 'V25BodyTempMax',
#     'V26PODduration', 'V27PODIV', 'V28PODPO'
# ]]
# y = df1[['V00GT']]  # dependent variable : target vector
# # ---------------------------------------------------------------------------


'--------------------- load model --------------------------- '
# ==================================================================
target_dir = './result_1317_RFE_test00_AUC95_PR46'  # 0-th elimination
# target_dir = './result_RFE_test01_AUC62'     # 1-th elimination        # rfe_1 Skin Flap vs GI flap_0
# ==================================================================


from keras.models import model_from_json
import pandas as pd

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
for seq_id in range(numvars): ## one-hot의 경우 median을 사용하지 않는다.
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
        X_sens_test = np.tile(x_sens_base, (Sens_test_num, 1))  # Make test base # augmentation
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

df_effect0.to_csv(target_dir + "rfe" + '.txt')

elim_idx, elim_pred_diff = min(enumerate(df_effect['mean']), key=lambda x: x[1])

elim_fName = X_total.columns[elim_idx]
print("Elim target : ", "#", elim_idx, " ", elim_fName, "w/ effect of ", elim_pred_diff)
print(df_effect2)

plt.plot(Pred_rx1[:, elim_idx], label='Sensitivity'.format(elim_pred_diff, elim_fName))
plt.title('Sensitivity of {}'.format(elim_fName))
plt.ylabel("Probability Effect")
plt.xlabel("Test Stage")
plt.legend(loc="best")

plt.show()