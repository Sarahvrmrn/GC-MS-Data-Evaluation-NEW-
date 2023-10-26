import pandas as pd
import numpy as np
from helpers import Helpers as hp
from datetime import datetime
from os.path import join
import os


def fill_RT(path: str, save_path: str, whatever:str):
    eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
    os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(join(save_path, f'fill_RT_{whatever}' + eval_ts))

    files = hp.get_all_files_in_dir_and_sub(path)
    files = [f for f in files if f.find('.csv') >= 0]
    for file in files:
        df = pd.read_csv(file, sep=',', decimal='.')
        df.drop(['RT(minutes) - NOT USED BY IMPORT', 'RI'],
                inplace=True, axis=1)
        df.set_index('RT(milliseconds)', inplace=True)
        new_index = np.arange(120000, 832200, 100)
        df = df.reindex(new_index, fill_value=0)

        name = file[file.rfind('\\')+1:file.rfind('.')]

        hp.save_df(df, join(os.environ["ROOT_PATH"]), name)


if __name__ == '__main__':
    print('start')
    save_path = 'C:\\Users\\sverme-adm\\Desktop\\fill_RT'
    for i in ['gesund', 'infiziert', 'Raum']:
        path = f"C:\\Users\\sverme-adm\\Desktop\\inf_ges\\{i}"
        fill_RT(path, save_path, i)
    print('finished')
