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
from scipy.signal import correlate
from scipy.interpolate import interp1d



def fix_shift(path: str, save_path: str, whatever:str):
    
    eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
    os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(join(save_path, f'fill_RT_{whatever}' + eval_ts))
    
    files = hp.get_all_files_in_dir_and_sub(path)
    files = [f for f in files if f.find('.csv') >= 0]
    
    time_interval = 100  # Replace with your actual time interval

    # Load and process reference chromatogram (you can adjust this part)
    reference_chromatogram = pd.read_csv('C:\\Users\\sverme-adm\\Desktop\\GitHub\\eval_phd_lollol\\20.04.2023_2023-04-20 inf Pflanze 90ml$min 60 min_02.csv', sep=';'
                                , decimal=',') # Reference chromatogram data

    reference_retention_times = reference_chromatogram['RT(milliseconds)'].values
    reference_intensity = reference_chromatogram['TIC'].values
 
    for file in files:
        df = hp.read_file(file, dec=',', sepi=';')[['RT(milliseconds)', 'TIC']]
        df.set_index('RT(milliseconds)', inplace=True)# Time interval between data points in chromatogram
        

            # Load the target chromatogram data and retention times
        target_chromatogram = df  # Replace with your loading logic
        target_retention_times = df.index.values
        target_intensity = target_chromatogram['TIC'].values# Replace with your loading logic

        # Calculate cross-correlation between reference and target intensities
        correlation = correlate(reference_intensity, target_intensity, mode='same')

        # Find the shift between chromatograms
        shift = (np.argmax(correlation) - len(reference_intensity) / 2) * time_interval

        # Apply the calculated shift to the target chromatogram's retention times
        adjusted_retention_times = target_retention_times - shift

        # Interpolate the target intensity onto the adjusted retention times
        f = interp1d(target_retention_times, target_intensity, kind='cubic', fill_value='extrapolate')
        adjusted_intensity = f(adjusted_retention_times)

            # Save the aligned chromatogram to the output folder
        aligned_data = pd.DataFrame({'RT(milliseconds)': adjusted_retention_times, 'TIC': adjusted_intensity})
        name = file[file.rfind('\\')+1:file.rfind('.')]
        
        hp.save_df(aligned_data, join(os.environ["ROOT_PATH"]), name, index=False)

    print("Alignment and saving complete.")


if __name__ == '__main__':
    print('start')
    save_path = 'C:\\Users\\sverme-adm\\Desktop\\shifted'
    for i in ['gesund', 'infiziert', 'Raum']:
        path = f"C:\\Users\\sverme-adm\\Desktop\\fill_RT\\{i}"
        fix_shift(path, save_path, i)
    print('finished')
