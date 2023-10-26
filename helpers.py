import json as js
from os import listdir,  scandir
from os.path import isfile, join
from pathlib import Path
from shutil import rmtree
import pandas as pd
import os


class Helpers:

    def concat_df(list_dfs:list[pd.DataFrame]):
        time_axis = list_dfs['time']
        print(time_axis)
        df = pd.concat(list_dfs, axis=1)
        # df.set_index(df.columns[0],inplace=True)
        df.drop(columns='time', inplace=True)
        df.set_index(time_axis, inplace=True)
        return df
    
    def round_data(data:pd.DataFrame, name:str):
        df_round  = data.round(2)
        df_round.set_index('Ret.Time', inplace=True)
        data =  []
        idx_new =[]
        for idx in df_round.index.unique():
           mean = df_round[df_round.index == idx].mean()[0]
           data.append(mean)
           idx_new.append(idx)
        df = pd.DataFrame({'time':idx_new, name:data})
        return df
        
        
        
    
    def df_to_dict(df: pd.DataFrame) -> dict:
        data_expanded = {}
        for col in df.columns:
            for i in df.index:
                data_expanded[f'p{i}_{col}'] = df.loc[i][col]
        return data_expanded

    def read_file(path: str, skip_header=0, dec='.', sepi=','):
        df = pd.read_csv(path,  sep=sepi, decimal=dec, skiprows=skip_header)
        return df

    def get_all_files_in_dir_and_sub(path: str):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))
        return files

    def get_all_folders(path):
        return [os.path.join(path, i) for i in os.listdir(path)]

    def save_html(html_object, path: str, name: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + name + '.html'
        print(path)
        html_object.write_html(path)

    def get_all_files_in_sub(path: str):
        return [join(path, f)
                for f in listdir(path) if isfile(join(path, f))]

    def get_key_by_value(data: dict, value: int):
        for key, val in data.items():
            if val == value:
                return key

    def sample_to_numbers(samlpes: pd.Series):
        samles_unique = samlpes.unique()
        sample_dict = {}
        for sample, i in zip(samles_unique, range(len(samles_unique))):
            sample_dict[sample] = i
        numbers = [sample_dict[s] for s in samlpes]
        return sample_dict, numbers

    def read_json(folder, filename):
        # read json to dict
        with open(join(folder, filename)) as json_file:
            return js.load(json_file)

    def get_name_from_info(info: dict):
        # extract name from path
        name = f"{info['sample']}_{info['height']}_{info['number']}"
        return name

    def get_subfolders(path):
        # list all subfolders
        return [f.path for f in scandir(path) if f.is_dir()]

    def clean_info_meaurement(info: dict):
        # deletes obsolete infos about measurement
        cleaned_info = {}
        for key in ['datetime', 'height', 'number', 'rate', 'sample']:
            cleaned_info[key] = info[key]
        return cleaned_info

    def get_path_data(path):
        # returns path to txt with data
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('txt') > 0:
                return join(path, file)

    def del_results(path):
        folders = [f.path for f in scandir(path) if f.is_dir()]
        for folder in folders:
            if folder.find('results') > 0:
                rmtree(join(path, folder))

    def one_layer_back(path: str) -> str:
        # get path one layer up
        new_path = path[:path.rfind('\\')]
        return new_path

    def get_path_info(path):
        # returns path to json with info
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('json') > 0:
                return join(path, file)

    def mkdir_ifnotexits(path):
        # mkdir if not exists
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def save_df(df, path, name, index=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = join(path, f'{name}.csv')
        print(path)
        df.to_csv(path, sep=';', decimal=',', index=index)
