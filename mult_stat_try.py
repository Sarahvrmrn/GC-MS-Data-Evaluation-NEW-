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


def doPCA(df: pd.DataFrame, df_info: pd.DataFrame):
    
    components_PCA = 100
    pca = PCA(n_components=components_PCA).fit(df)
    variance_ratio = pca.explained_variance_ratio_

    # Print the percentage of variance explained by each PC
    for i, ratio in enumerate(variance_ratio):
        print(f"PC{i + 1}: {ratio * 100:.2f}%")
    
    total_variance = np.sum(variance_ratio)

# Print the sum of all the percentages
    print(f"Total variance explained: {total_variance * 100:.2f}%")
        
    principalComponents = pca.transform(df)
    
    dfPCA = pd.DataFrame(data=principalComponents, columns=[
                         f'PC{i+1}' for i in range(components_PCA)])
    dfPCA['label'] = df_info['name']
    print(dfPCA)
    #fig = px.scatter_3d(dfPCA, x='PC1', y='PC2', z='PC3',
    #                    color='label', hover_data=df_info.to_dict('series'))
    #hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    #fig.show()
    
    return dfPCA


def doLDA(df_train: pd.DataFrame, df_info: pd.DataFrame):
    
    # Performing LDA and testing Fitting
    X_train = df_train
    y_train = df_info['name']
    X_test = df_test
    
    
    lda =LDA()
    lda.fit(X_train, y_train)
    X_train_transformed = lda.transform(X_train)
    X_test_transformed = lda.transform(X_test)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# Plot the training data
    ax.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], X_train_transformed[:, 2], c=y_train, cmap='viridis', label='Training Data')

# Plot the test data
    ax.scatter(X_test_transformed[:, 0], X_test_transformed[:, 1], X_test_transformed[:, 2], c='red', marker='x', label='Test Data')

# Set labels and legend
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    ax.legend()

# Show the 3D plot
    plt.show()
    


def doRFC(df: pd.DataFrame, df_info: pd.DataFrame):
    
    # Performing LDA and testing Fitting
    X = df
    y = df_info['name']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, random_state=0)
    
    model = RandomForestClassifier()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)