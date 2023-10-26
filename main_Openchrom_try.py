import pandas as pd
import os
from mult_stat import doPCA, doLDA, doRFC
from helpers import Helpers as hp
from os.path import join
from datetime import datetime
from time import sleep
path_train = 'C:\\Users\\sverme-adm\\Desktop\\PreProcessing2'
path_test = 'C:\\Users\\sverme-adm\\Desktop\\PreProcessingTest'
save_path = 'C:\\Users\\sverme-adm\\Desktop\\resultsPreProcessing'
eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(join(save_path, 'result' + eval_ts))

files_train = hp.get_all_files_in_dir_and_sub(path_train)
files = [f for f in files_train if f.find('.csv') >= 0]

files_test = hp.get_all_files_in_dir_and_sub(path_test)
files = [f for f in files_test if f.find('.csv') >= 0]

merged_df_train = pd.DataFrame()
merged_df_test = pd.DataFrame()
info_train = []
info_test = []



for file in files_train:
    df_train = hp.read_file(file )[['RT(milliseconds)', 'TIC' ]]
    df_train.set_index('RT(milliseconds)', inplace=True)
    merged_df_train = pd.merge(merged_df_train, df_train, how='outer', left_index=True, right_index=True)
    
    merged_df_train = merged_df_train.fillna(0)
    merged_df_train = merged_df_train.rename(columns={'TIC': file.split('\\')[5]})

    info_train.append({'name': file.split('\\')[5], 'filename': os.path.basename(file)})
    
    
for file in files_test:
    df_test = hp.read_file(file )[['RT(milliseconds)', 'TIC' ]]
    df_test.set_index('RT(milliseconds)', inplace=True)
    merged_df_test = pd.merge(merged_df_test, df_test, how='outer', left_index=True, right_index=True)
    
    merged_df_test = merged_df_test.fillna(0)
    merged_df_test = merged_df_test.rename(columns={'TIC': file.split('\\')[5]})

    info_test.append({'name': file.split('\\')[5], 'filename': os.path.basename(file)})

df_info_train = pd.DataFrame(info_train)
df_info_test = pd.DataFrame(info_test)
#merged_df.drop(merged_df.index[:25], inplace=True)
#print(df_info)
print(merged_df_test)


hp.save_df(merged_df_train, join(os.environ["ROOT_PATH"], 'data'), 'extracted_features')
hp.save_df(df_info_train, join(
    os.environ["ROOT_PATH"], 'data'), 'extracted_features_info')
dfPCA_train = doPCA(merged_df_train, df_info_train)

dfPCA_train = dfPCA_train.drop('label', axis=1)

print(dfPCA_train)


dfLDA = doLDA(dfPCA_train, df_info_train)

#LDA_test = doLDA(dfLDA, merged_df_test)
#dfRFC = doRFC(merged_df.T, df_info)

