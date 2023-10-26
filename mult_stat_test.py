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
from os.path import join
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def do_statistic(df: pd.DataFrame, df_info: pd.DataFrame):
    
    components_PCA = 3
    pca = PCA(n_components=components_PCA).fit(df)
    principalComponents = pca.transform(df)
    
    dfPCA = pd.DataFrame(data=principalComponents, columns=[
                         f'PC{i+1}' for i in range(components_PCA)])
    dfPCA['label'] = df_info['name']
    print(dfPCA)
    fig = px.scatter_3d(dfPCA, x='PC1', y='PC2', z='PC3',
                        color='label', hover_data=df_info.to_dict('series'))
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    fig.show()
    
    components_LDA = 3
    lda = LDA(n_components=components_LDA)
    lda.fit(principalComponents, dfPCA['label'])
    data_lda = lda.transform(principalComponents)
    lda_components = lda.scalings_
    class_means = lda.means_
    print(lda_components)
    print(class_means)
    dfLDA = pd.DataFrame(data=data_lda, columns=[
                         f'LD{i+1}' for i in range(components_LDA)])
    dfLDA['label'] = df_info['name']
    print(dfLDA)
    
    fig = px.scatter_3d(dfLDA, x='LD1', y='LD2', z = 'LD3',
                        color='label', hover_data=df_info.to_dict('series'))
    fig.show()
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')

    num_folds = 10
    
    scores = cross_val_score(lda, principalComponents, dfPCA['label'], cv = num_folds)
    print('cross-validation scores:', scores)
    print( 'mean score:', scores.mean())    print('Standard deviation:', scores.std())
    
    return dfPCA, dfLDA


def doLDA(df: pd.DataFrame, df_info: pd.DataFrame):
    
    X = df
    y = df_info['name']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    lda =LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    components_LDA = 3
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    y = df_info['name']
    lda = LDA(n_components=components_LDA)
    LinearComponents = lda.fit_transform(df_scaled, y) 
    
    scores =cross_val_score(lda, df_scaled, y, cv=5)
    


    # Generate a confusion matrix
    y_pred = lda.predict(df_scaled)
    cm = confusion_matrix(y, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    print(cm)
    top_margin =0.06
    bottom_margin = 0.06
    
    
    dfLDA = pd.DataFrame(data=LinearComponents, columns=[
                         f'LD{i+1}' for i in range(components_LDA)])
    dfLDA['label'] = df_info['name']
    print(dfLDA)
    
    fig = px.scatter_3d(dfLDA, x='LD1', y='LD2', z = 'LD3',
                        color='label', hover_data=df_info.to_dict('series'))
    fig.show()
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')

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
    
    return dfLDA, LDA


def loo_cv(df: pd.DataFrame, df_info: pd.DataFrame) -> float:
    result = []
    lda = LDA(n_components=3)
    for i in range(len(df.index)):


        df_test = df.iloc[i]
        label_test = df_info.iloc[i]

        df_train = np.array(df.drop(df.index[i]))
        label_train = df_info.drop(i) #################################
        ######################################
        ###############################
        # ['name'].to_list()

        lda.fit_transform(df_train, label_train)

        prediction = lda.predict([df_test])[0]
        result.append([prediction, label_test['name']])

    df = pd.DataFrame(result, columns=['pred', 'sample'])
    df['true_pred'] = df['pred'] == df['sample']
    accuracy = (df.true_pred.value_counts()[True]/len(df))*100
    print(f'accuracy of loo_cv is {accuracy}%')
    cm = confusion_matrix(df['sample'], df['pred'])
    print(cm)


