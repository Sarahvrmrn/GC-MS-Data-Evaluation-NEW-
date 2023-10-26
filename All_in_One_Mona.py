import pandas as pd
import os
import seaborn as sns
from helpers import Helpers as hp
from os.path import join
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean



# Get all files for the training data set and merge them in on DataFrame

path_train = 'C:\\Users\\sverme-adm\\Desktop\\Daten_Mona'
save_path_train = 'C:\\Users\\sverme-adm\\Desktop\\results_DM'

path_test = 'C:\\Users\\sverme-adm\\Desktop\\Daten_Mona_Test'
save_path_test = 'C:\\Users\\sverme-adm\\Desktop\\results_DMT'

eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(
    join(save_path_train, 'result' + eval_ts))

components_LDA = 2
components_PCA = 30


def read_files(path: str, tag: str):
    files = hp.get_all_files_in_dir_and_sub(path)
    files = [f for f in files if f.find('.csv') >= 0]
    merged_df_train = pd.DataFrame()
    info_train = []

    for file in files:
        df = hp.read_file(file)[['RT(milliseconds)', 'TIC']]
        df.set_index('RT(milliseconds)', inplace=True)

        
        merged_df_train = pd.merge(
            merged_df_train, df, how='outer', left_index=True, right_index=True)

        merged_df_train = merged_df_train.fillna(0)
        merged_df_train = merged_df_train.rename(
            columns={'TIC': file.split('\\')[5]})

        info_train.append(
            {'Class': file.split('\\')[5], 'filename': os.path.basename(file)})

    df_info_train = pd.DataFrame(info_train)
    merged_df_train.drop(merged_df_train.index[:25], inplace=True)
    hp.save_df(merged_df_train, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_{tag}')
    hp.save_df(df_info_train, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_info_{tag}')
    
    return merged_df_train
    
    

    # Perform Data Reduction with PCA on the DataFrame


def create_pca(path_merged_data_train: str, path_merged_data_train_info: str):
    df = pd.read_csv(path_merged_data_train, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_train_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    pca = PCA(n_components=components_PCA).fit(df.T)

    principalComponents = pca.transform(df.T)

    df_PCA = pd.DataFrame(data=principalComponents, columns=[
        f'PC{i+1}' for i in range(components_PCA)])
    df_PCA.set_index(df_info['Class'], inplace=True)

    return pca, df_PCA


def create_lda(df_pca: pd.DataFrame):
    X = df_pca
    y = df_pca.index
    lda = LDA(n_components=components_LDA).fit(X, y)
    X_lda = lda.fit_transform(X.values, y)
    dfLDA_train = pd.DataFrame(data=X_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    dfLDA_train.index = y
    
    
    return lda, dfLDA_train






def push_to_pca(pca: PCA, path_merged_data_test: str, path_merged_data_test_info: str):
    df = pd.read_csv(path_merged_data_test, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_test_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    transformed_data = pca.transform(df.T)
    dfPCA_test = pd.DataFrame(data=transformed_data, columns=[
        f'PC{i+1}' for i in range(components_PCA)], index=df_info['Class'])
    return dfPCA_test


def push_to_lda(lda: LDA, transformed_data: pd.DataFrame):
    predictions = lda.predict(transformed_data.values)
    transformed_data_lda = lda.transform(transformed_data.values)
    df_lda_test_transformed = pd.DataFrame(data=transformed_data_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)], index=transformed_data.index)
    return df_lda_test_transformed, predictions


def combine_data(df_test: pd.DataFrame, df_train: pd.DataFrame):
    df_test['Dataset'] = ['test' for _ in df_test.index]
    df_train['Dataset'] = ['train' for _ in df_train.index]
    df_merged = pd.concat([df_train, df_test], ignore_index=False, sort=False)
    return df_merged


def plot(df: pd.DataFrame):
    fig = px.scatter(df, x='LD1', y='LD2', color=df.index, symbol='Dataset', symbol_sequence= ['circle', 'x'])
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    fig.show()
    


if __name__ == '__main__':
    df_raw_train = read_files(path_train, 'train')
    print(df_raw_train)
    exit()
    df_raw_test = read_files(path_test, 'test')

    pca, df_pca = create_pca(join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_train.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_train.csv')
    )
    lda, df_lda_train = create_lda(df_pca)
    transformded_data_test = push_to_pca(pca, join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_test.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_test.csv'))
    df_lda_test, predictions = push_to_lda(lda, transformded_data_test)
    merged_df = combine_data(df_lda_test, df_lda_train)
    plot(merged_df)
    
    # Calculate accuracy for each class
    accuracy_per_class = {}
    unique_classes = set(df_lda_test.index)  # Get the unique classes in the test dataset

    for cls in unique_classes:
        indices = df_lda_test.index == cls  # Filter test labels for the current class
        accuracy = accuracy_score(df_lda_test.index[indices], predictions[indices])  # Calculate accuracy for the current class
        accuracy_per_class[cls] = accuracy

    # Print accuracy for each class
    for cls, accuracy in accuracy_per_class.items():
        print(f"Accuracy for class {cls}: {accuracy:.2f}")
