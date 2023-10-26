import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helpers import Helpers as hp
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from os.path import join
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def doPCA(df: pd.DataFrame, df_info: pd.DataFrame):
    
    components_PCA = 100
    pca = PCA(n_components=components_PCA).fit(df.T)
    variance_ratio = pca.explained_variance_ratio_

    # Print the percentage of variance explained by each PC
    for i, ratio in enumerate(variance_ratio):
        print(f"PC{i + 1}: {ratio * 100:.2f}%")
    
    total_variance = np.sum(variance_ratio)

# Print the sum of all the percentages
    print(f"Total variance explained: {total_variance * 100:.2f}%")
        
    principalComponents = pca.transform(df.T)
    
    dfPCA = pd.DataFrame(data=principalComponents, columns=[
                         f'PC{i+1}' for i in range(components_PCA)])
    dfPCA['label'] = df_info['name']
    print(dfPCA)
    
    return dfPCA
    
def doNN(df: pd.DataFrame, df_info:pd.DataFrame):  
      
    X = df
    y = df_info['name']
    
    n_components = 100
    num_classes = 6
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_components))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=5)
    loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions
    X_new_data = model
    predictions = model.predict(X_new_data)
    
    print(loss, accuracy, predictions)
    #fig = px.scatter_3d(dfPCA, x='PC1', y='PC2', z='PC3',
    #                    color='label', hover_data=df_info.to_dict('series'))
    #hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    #fig.show()
    
    


