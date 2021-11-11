import random
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
# from pyclustering.cluster.clarans import clarans;
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
from sklearn import datasets
from pyclustering.cluster import cluster_visualizer_multidim

df = pd.read_csv('D:/housing.csv', delimiter=",")
df_original = df.copy()

print(df.shape)
print(df.isnull().sum())

housing_corr_matrix = df.corr()
# set the matplotlib figure
fig, axe = plt.subplots(figsize=(12, 8))
# Generate color palettes
cmap = sns.diverging_palette(200, 10, center="light", as_cmap=True)
# draw the heatmap
sns.heatmap(housing_corr_matrix, vmax=1, square=True, cmap=cmap, annot=True)
plt.show()

encoders = [LabelEncoder(), OneHotEncoder()]
scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
models = ['K_Means', 'DBSCAN', 'CLARANS', 'MeanShift', 'GMM']
hyperparams = {
    # 'K_Means_params':{}
    # 'GMM_params':{}
    # 'CLARANS_params':{}
    'DBSCAN_params': {
        'eps': [0.005, 0.01],
        'min_samples': [5, 10]
        # 'eps':[0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'MeanShift_params': {
        'n': [10, 15, 20]
    },
    'k': range(2, 9)
}


def preprocessing(df):
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

    return df


def main(df, y, scalers, models, hyperparams, combi):
    new_df = preprocessing(df)
    for i in combi:
        X = new_df[i]

        print("Current combination", i)

        for scaler in scalers:
            print("Current scaler:", scaler)
            scaled_X = scaler.fit_transform(X)
            data_df = pd.DataFrame(scaled_X)
            clustering(data_df, y, models, hyperparams)


def elbow_curve(distortions):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 9), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    # plt.show()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering(df, y, models, hyperparams):
    # Experiment with various models
    for model in models:
        print("Current model: ", model)
        # Apply various hyperparameters in each models
        if model == 'K_Means':
            distortions = []
            for k in hyperparams['k']:
                kmeans = KMeans(n_clusters=k, init='k-means++')
                cluster = kmeans.fit(df)
                labels = kmeans.predict(df)
                cluster_id = pd.DataFrame(cluster.labels_)
                distortions.append(kmeans.inertia_)

                d1 = pd.concat([df, cluster_id], axis=1)
                d1.columns = [0, 1, "cluster"]

                sns.scatterplot(d1[0], d1[1], hue=d1['cluster'], legend="full")
                sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], label='Centroids')
                plt.title("KMeans Clustering")
                plt.legend()
                plt.show()

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " ", k,
                      "-clusters)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))

                print('Quantile comparison score(purity_score):', purity_score(y, labels))

            elbow_curve(distortions)



        elif model == 'GMM':
            for k in hyperparams['k']:
                gmm = GaussianMixture(n_components=k)
                gmm.fit(df)
                labels = gmm.predict(df)

                frame = pd.DataFrame(df)
                frame['cluster'] = labels
                frame.columns = [df.columns[0], df.columns[1], 'cluster']

                for i in range(0, k + 1):
                    data = frame[frame["cluster"] == i]
                    plt.scatter(data[data.columns[0]], data[data.columns[1]])
                plt.show()

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " (", k,
                      "-components)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))

                print('Quantile comparison score(purity_score):', purity_score(y, labels))


        elif model == 'CLARANS':
            data = df.values.tolist()
            for k in hyperparams['k']:
                cl_data = random.sample(data, 250)
                clarans_obj = clarans(cl_data, k, 3, 5)
                (tks, res) = timedcall(clarans_obj.process)
                clst = clarans_obj.get_clusters()
                med = clarans_obj.get_medoids()

                # print("Index of clusters' points :\n", clst)
                # print("\nIndex of the best medoids : ", med)

                labels = pd.DataFrame(clst).T.melt(var_name='clusters').dropna()
                labels['value'] = labels.value.astype(int)
                labels = labels.sort_values(['value']).set_index('value').values.flatten()

                vis = cluster_visualizer_multidim()
                vis.append_clusters(clst, cl_data, marker="*", markersize=5)
                vis.show(max_row_size=3)

                print('Silhouette Score(euclidean):', metrics.silhouette_score(cl_data, labels, metric='euclidean'),
                      " (", k, "-clusters)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(cl_data, labels, metric='manhattan'))





        elif model == 'DBSCAN':
            eps = hyperparams['DBSCAN_params']['eps']
            minsam = hyperparams['DBSCAN_params']['min_samples']

            for i in eps:
                for j in minsam:
                    db = DBSCAN(eps=i, min_samples=j)
                    cluster = db.fit(df)
                    cluster_id = pd.DataFrame(cluster.labels_)

                    d2 = pd.DataFrame()
                    d2 = pd.concat([df, cluster_id], axis=1)
                    d2.columns = [0, 1, "cluster"]

                    sns.scatterplot(d2[0], d2[1], hue=d2['cluster'], legend="full")
                    plt.title('DBSCAN with eps {}'.format(i))
                    plt.show()

                    print('Silhouette Score(euclidean):',
                          metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='euclidean'), " (eps=", i,
                          ")", " (min_samples=", j, ")")
                    print('Silhouette Score(manhattan):',
                          metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='manhattan'))



        elif model == 'MeanShift':
            n = hyperparams['MeanShift_params']['n']
            for i in n:
                bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=i)
                ms = MeanShift(bandwidth=bandwidth)
                cluster = ms.fit(df)
                cluster_id = pd.DataFrame(cluster.labels_)

                d6 = pd.DataFrame()
                d6 = pd.concat([df, cluster_id], axis=1)
                d6.columns = [0, 1, "cluster"]

                sns.scatterplot(d6[0], d6[1], hue=d6['cluster'], legend="full")
                plt.title('Mean Shift with {} samples'.format(i))
                plt.show()

                print('n_samples(estimate_bandwidth) = {}'.format(i))
                print('Silhouette Coefficient(euclidean): ',
                      metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='euclidean'))
                print('Silhouette Coefficient(manhattan): ',
                      metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='manhattan'))


combi = []
combi.append(['longitude', 'latitude'])
# combi.append(['longitude', 'latitude', 'population'])
combi.append(['total_rooms', 'total_bedrooms'])
combi.append(['population', 'households'])

quantiles = list(df['median_house_value'].quantile([0.25, 0.5, 0.75, 1.0]))
df.loc[df['median_house_value'] >= quantiles[0], 'quantiles'] = 1
df.loc[df['median_house_value'] >= quantiles[1], 'quantiles'] = 2
df.loc[df['median_house_value'] >= quantiles[2], 'quantiles'] = 3
df.loc[df['median_house_value'] >= quantiles[3], 'quantiles'] = 4

y = df['quantiles'].astype("category")

main(df, y, scalers, models, hyperparams, combi)
