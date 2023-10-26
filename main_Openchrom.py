import pandas as pd
import os
from mult_stat import doPCA,  doLDA
from helpers import Helpers as hp
from os.path import join
from datetime import datetime
from time import sleep
path = 'C:\\Users\\sverme-adm\\Desktop\\inf_ges'
save_path = 'C:\\Users\\sverme-adm\\Desktop\\results_inf_ges'
eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(join(save_path, 'result' + eval_ts))

files = hp.get_all_files_in_dir_and_sub(path)
files = [f for f in files if f.find('.csv') >= 0]

merged_df = pd.DataFrame()
info = []



for file in files:
    df = hp.read_file(file )[['RT(milliseconds)', 'TIC' ]]
    df.set_index('RT(milliseconds)', inplace=True)
    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
    
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.rename(columns={'TIC': file.split('\\')[5]})

    info.append({'name': file.split('\\')[5], 'filename': os.path.basename(file)})

df_info = pd.DataFrame(info)
#merged_df.drop(merged_df.index[:25], inplace=True)
#print(df_info)
print(merged_df)

hp.save_df(merged_df, join(os.environ["ROOT_PATH"], 'data'), 'extracted_features')
hp.save_df(df_info, join(
    os.environ["ROOT_PATH"], 'data'), 'extracted_features_info')
dfPCA = doPCA(merged_df, df_info)
dfPCA = dfPCA.drop('label', axis=1)
#print(dfPCA)
dfLDA = doLDA(dfPCA, df_info)
print(dfLDA)

#dfRFC = doRFC(merged_df.T, df_info)

