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
    
    components_PCA = 65
    pca = PCA(n_components=components_PCA).fit(df.T)
    variance_ratio = pca.explained_variance_ratio_

    # Print the percentage of variance explained by each PC
    for i, ratio in enumerate(variance_ratio):
        print(f"PC{i + 1}: {ratio * 100:.2f}%")
    
    total_variance = np.sum(variance_ratio)


# Print the sum of all the percentages
    print(f"Total variance explained: {total_variance * 100:.2f}%")
    
    loadings = pca.components_
    print(loadings)
        
    principalComponents = pca.transform(df.T)
    
    dfPCA = pd.DataFrame(data=principalComponents, columns=[
                         f'PC{i+1}' for i in range(components_PCA)])
    dfPCA['label'] = df_info['name']
    print(dfPCA)
    #fig = px.scatter_3d(dfPCA, x='PC1', y='PC2', z='PC3',
    #                    color='label', hover_data=df_info.to_dict('series'))
    #hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    #fig.show()
    
    return dfPCA


def doLDA(df: pd.DataFrame, df_info: pd.DataFrame):
    
    # Performing LDA and testing Fitting
    X = df
    y = df_info['name']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    components_LDA = 2
    
    lda =LDA(n_components=components_LDA).fit(X,y)
    y_pred = lda.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    hp.save_df(report_df, join(os.environ["ROOT_PATH"], 'data'), 'classification_report')
    
    
    # Plot LDA
    X_lda =lda.fit_transform(X,y)
    dfLDA = pd.DataFrame(data=X_lda, columns=[
                         f'LD{i+1}' for i in range(components_LDA)])
    dfLDA['label'] = df_info['name']
    print(dfLDA)
    
    fig = px.scatter(dfLDA, x='LD1', y='LD2',
                        color='label', hover_data=df_info.to_dict('series'))
    fig.show()
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    
 
    # Generate a confusion matrix
    y_pred = lda.predict(df)
    cm = confusion_matrix(y, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    print(cm)
    top_margin =0.06
    bottom_margin = 0.06
    
    # Plot the confusion matrix as a heatmap using Seaborn
    fig, ax = plt.subplots(
        figsize=(10,8), 
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))
    sns.heatmap(cm / cm_sum.astype(float), annot=True , cmap='gist_earth', fmt='.2%')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(df_info['name'].unique().tolist(), fontsize=5.5)
    ax.yaxis.set_ticklabels(df_info['name'].unique().tolist(), fontsize=5.5)
    plt.show()
    
    return dfLDA

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