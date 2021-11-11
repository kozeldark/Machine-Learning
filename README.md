# Machine-Learning

**Wisconsin breast cancer Prediction(PHW 1)**

* Import
``` Python
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,classification_report,roc_curve,plot_roc_curve,auc,precision_recall_curve,plot_precision_recall_curve,average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

```

* Reads the dataset and generates a data frame.
``` Python
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape','marginal_adhesion',  'single_epithelial_size', 'bare_nuclei',
         'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",sep=',',names=names).replace('?', np.nan).dropna()
print(df.describe())
print(df.isna().sum())

```
![image](https://user-images.githubusercontent.com/77625823/141355323-2bfcf727-d38d-43fc-bd9f-b1acb03cc595.png)

* A function that will show the correlation of each feature.
``` Python
def visualizationCorrelation(df):
    # compute the corr matrix
    corr = df.corr()

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 6))

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # draw the heatpmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})
    plt.subplots_adjust(left=0, bottom=0.24, right=1, top=1)
    plt.show()
```
![image](https://user-images.githubusercontent.com/77625823/141355409-b0357475-4b1b-4b30-bdc5-cbb1e9173ee5.png)

* Since we saw that id values are irrelevant, drop id.
``` Python
df.drop(['id'], 1, inplace=True)
```

* Divide the features and target and split the dataset.
``` Python
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
```

* Models & Scalers that have undergone hyperparameter tuning to be used in grid search.
``` Python
g_models = [
            (DecisionTreeClassifier(), [{'criterion':['entropy'],'splitter':['best','random'],
                                        'max_depth':[None,2,3],'max_features':[None, 'sqrt','log2']}]),
            (DecisionTreeClassifier(), [{'criterion':['gini'],'splitter':['best','random'],
                                        'max_depth':[None,2,3],'max_features':[None, 'sqrt','log2']}]),
            (LogisticRegression(), [{'C' : [0.001, 0.01, 0.1, 1, 10, 100],'max_iter': [100,1000]}]),
            (SVC(), [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}])]

# Scaler list
scaler = [StandardScaler(), MinMaxScaler(),MaxAbsScaler(),RobustScaler()]

# k-fold k value list
k_fold_k = [5, 6, 7, 8, 9, 10]
```

* A function that will find the best combination.
``` Python
def findbest(scaler,g_models):
    # Proceed with hyperparameter tuning through grid search.
    for scaler in scaler:
        X_train_res = scaler.fit_transform(X_train)
        X_test_res = scaler.fit_transform(X_test)
        result_list = []
        for model, param in g_models:
            for k in k_fold_k:
                result = []
                grid = GridSearchCV(estimator=model, param_grid=param, scoring='accuracy', cv=k, n_jobs=-1)
                grid.fit(X_train_res, y_train)
                #print(' {}: \n Best Accuracy: {:.2f} %'.format(model, grid.best_score_ * 100))
                #print('\n Best Parameter : {}', grid.best_params_)

                # predict with best model and calculate MSE
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test_res)

                #print(confusion_matrix(y_test, y_pred))
                #print(classification_report(y_test, y_pred))

                result.append(model)
                result.append(grid.best_params_)
                result.append(scaler)
                result.append(accuracy_score(y_test, y_pred) * 100)
                result.append(k)
                result_list.append(result)
                result_df = pd.DataFrame(result_list, columns=['Model', 'Best parameters', 'Scaler', 'Accuracy', 'k-fold k value'])
        print(result_df)
        print()
```

![image](https://user-images.githubusercontent.com/77625823/141355632-ef30b963-d825-4a4e-b33f-27245f8c6e55.png)
![image](https://user-images.githubusercontent.com/77625823/141355640-2cdf1ede-81bb-4142-bafe-ca543d0a3e1b.png)
![image](https://user-images.githubusercontent.com/77625823/141355734-80c7b84b-9b22-47c2-8ae8-416e1a7d0271.png)
![image](https://user-images.githubusercontent.com/77625823/141355657-d1cae0f3-b37f-445b-b6da-0f689485d454.png)



**#################################################################################**


**California Housing Prices Clustering(PHW 2)**

* Before run this manual, please make sure the install and import following packages.
```Python
!pip install pyclustering
```

* Import
```Python
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
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
```

* Loading a Dataset
```Python
#Loading a dataset
df = pd.read_csv('/content/drive/MyDrive/housing.csv', delimiter = ",")
df_original = df.copy()

print(df.shape)
print(df.isnull().sum())
```
![image](https://user-images.githubusercontent.com/77625823/141356244-2b05c45c-afc3-416a-a3b0-56b1a5e1001a.png)

* Visualize it with a heat map.
```Python
housing_corr_matrix = df.corr()
#set the matplotlib figure
fig, axe = plt.subplots(figsize=(12,8))
#Generate color palettes
cmap = sns.diverging_palette(200, 10, center = "light", as_cmap=True)
#draw the heatmap
sns.heatmap(housing_corr_matrix, vmax=1, square =True, cmap=cmap, annot=True )
plt.show()
```

![image](https://user-images.githubusercontent.com/77625823/141356279-4e36db5e-02db-4608-b23d-db99f91942ec.png)


* Setting up the combinations.
You can freely add or delete the elements you want. 
```Python
encoders = [LabelEncoder(), OneHotEncoder()]
scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
models = ['K_Means','MeanShift','CLARANS','DBSCAN','GMM']
hyperparams = {
    #'K_Means_params':{}
    #'GMM_params':{}
    #'CLARANS_params':{}
    'DBSCAN_params': {
        'eps': [0.005, 0.01] 
        'min_samples': [10, 20]
    },
    'MeanShift_params': {
        'n': [10, 50, 100]
    },
    'k': range(2, 13)
}
```

* Various combinations of the features
We selected two features randomly and conducted the experiment.
```Python
combi = []
combi.append(['longitude', 'latitude'])
combi.append(['total_rooms', 'total_bedrooms'])
combi.append(['population','households'])
```
* Clean and prepare a dataset 

Function
```Python
preprocessing(df)
```

Parameters
```Python
df : dataset
```

```Python
'''
<  preprocessing  >
 Input : df

- Remove needless features.(median_house_value)
- fill the missing values

 Output : modified dataframe
'''


def preprocessing(df):
    df.drop(columns=["median_house_value"], inplace=True)
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

    return df
```

* Utils
Implement silhouette score function and elbow curve function
```Python
def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return silhouette_score(X, cluster_labels)
```

```Python
def elbow_curve(distortions):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 13), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()
```

* Clustering
We defined functions to cluster automatically with computing all combination of parameters that specified scaler, models and hyperparameters.

Function
```Python
clustering(df, models, hyperparams)
```

Parameters
```Python
df : dataset
models : list of models
['K_Means',' MeanShift', 'CLARANS', 'DBSCAN', 'GMM'] 
hyperparams : list of models’ hyperparameters
hyperparams = {
'DBSCAN_params': { 'eps': [0.01, 0.003] },
'MeanShift_params': { 'n': [10, 50, 100] },
'k': range(2, 13)
}
```

```Python
'''
<  clustering  >
 Input : df(dataframe), models(list), hyperparams(dict)

- clustering with various models and hyperparameter values.
- plotting

 Output : plot results, silhouette Score, Quantile comparison score

'''

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
                #plt.show()

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " ", k, "-clusters)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))
                #Compare the clustering results with N (where 2<= N <= 10) quantiles of the medianHouseValue feature values in the original dataset.
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

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " (", k, "-components)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))
             #Compare the clustering results with N (where 2<= N <= 10) quantiles of the medianHouseValue feature values in the original dataset.
                print('Quantile comparison score(purity_score):', purity_score(y, labels))


        elif model == 'CLARANS':
            data = df.values.tolist()
            for k in hyperparams['k']:
                cl_data = random.sample(data, 250)
                clarans_obj = clarans(cl_data, k, 3, 5)
                (tks, res) = timedcall(clarans_obj.process)
                clst = clarans_obj.get_clusters()
                med = clarans_obj.get_medoids()

                #print("Index of clusters' points :\n", clst)
                #print("\nIndex of the best medoids : ", med)

                labels = pd.DataFrame(clst).T.melt(var_name='clusters').dropna()
                labels['value'] = labels.value.astype(int)
                labels = labels.sort_values(['value']).set_index('value').values.flatten() 

                vis = cluster_visualizer_multidim()
                vis.append_clusters(clst, cl_data, marker="*", markersize=5)
                vis.show(max_row_size=3)

                print('Silhouette Score(euclidean):', metrics.silhouette_score(cl_data, labels, metric='euclidean'), " (", k, "-clusters)")
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

                    print('Silhouette Score(euclidean):', metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='euclidean'), " (eps=", i, ")", " (min_samples=", j, ")")
                    print('Silhouette Score(manhattan):', metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='manhattan'))



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
                print('Silhouette Coefficient(euclidean): ',metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='euclidean'))
                print('Silhouette Coefficient(manhattan): ',metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='manhattan'))
```

* Main
All of these processes are executed automatically by calling the main function.
```Python
'''
< main >

INPUT : df(dataframe), scalers(list), models(list), hyperparams(dict), combi(list)

- preprocessing
- scaling
- Call 'clustering()' function to cluster and plot

'''

def main(df, scalers, models, hyperparams, combi):
    new_df = preprocessing(df)
    for i in combi:
        X = new_df[i]

        print("Current combination", i)

        for scaler in scalers:
            print("Current scaler:", scaler)
            scaled_X = scaler.fit_transform(X)
            data_df = pd.DataFrame(scaled_X)
            clustering(data_df, models, hyperparams)
```

```Python
main(df, scalers, models, hyperparams, combi)
```

* Result
There are so many results that I'll show you one example.
If you want to see all the results, please refer to the attached 'Output Pictures.zip'.

_['longitude', 'latitude']_

*K-means

Best combination: MaxAbsScaler, k=2

![image](https://user-images.githubusercontent.com/77625823/141358389-3fa71c14-9f24-4a18-a484-5e3b7620c575.png)

![image](https://user-images.githubusercontent.com/77625823/141358400-e206e629-3eba-4d00-8c26-b680bb0d419f.png)

![image](https://user-images.githubusercontent.com/77625823/141358408-6e18fd41-36bc-4b8f-8d70-2ba99adf6a3e.png)


*MeanShift

Best combination: MaxAbsScaler, n_samples=15

![image](https://user-images.githubusercontent.com/77625823/141358513-d877a248-cf25-4242-b360-8c194689c085.png)

![image](https://user-images.githubusercontent.com/77625823/141358518-1dd5af22-ba0d-45dc-bc7e-3d589a88624b.png)


*CLARANS

Best combination: MaxAbsScaler, k=2

![image](https://user-images.githubusercontent.com/77625823/141358561-a2bed3a6-89cb-4bbb-8ae5-4c272abf0b6c.png)

![image](https://user-images.githubusercontent.com/77625823/141358567-dadef12f-87a4-4b7e-8762-4cf6d728ca34.png)


*DBSCAN

Best combination: MinMaxScaler, eps=0.005, min_samples=5

![image](https://user-images.githubusercontent.com/77625823/141358607-07c76e5b-df5d-461b-9c72-913b92c8ac88.png)

![image](https://user-images.githubusercontent.com/77625823/141358613-9a8d6b44-12db-4504-b1bf-5e6e0397752b.png)


*GMM

Best combination: MaxAbsScaler, k=12

![image](https://user-images.githubusercontent.com/77625823/141358650-988d269c-5f8f-4eda-9581-ec40bae51c57.png)

![image](https://user-images.githubusercontent.com/77625823/141358658-3791f6c6-9b6d-4f7b-a2a9-73613bf0f8fc.png)



*Compare the clustering results

Compare the results with N quantiles of the medianHouseValue feature values in the original dataset. In this case, we compared with N=4, N=5,and N=8.

There are so many results that only n=4 is shown as an example.

```Python
df = pd.read_csv('housing.csv', delimiter=",")
df_original = df.copy()

df=df.sort_values(by=['median_house_value'], axis=0)

median_house_value=df['median_house_value']

n=4
k=1
tmp=0

median_house_value_q=[]
total_rooms_q=[]
total_bedrooms_q=[]
longitude_q=[]
latitude_q=[]
household_q=[]
population_q=[]


while True:
  tmp=(1/n)*k
  if tmp>1:
    break
  print(tmp)
  q_v=df.quantile(tmp)
  print(q_v)
  median_house_value_q.append(q_v['median_house_value'])
  total_rooms_q.append(q_v['total_rooms'])
  total_bedrooms_q.append(q_v['total_bedrooms'])
  longitude_q.append(q_v['longitude'])
  latitude_q.append(q_v['latitude'])
  household_q.append(q_v['households'])
  population_q.append(q_v['population'])
  k=k+1


sns.scatterplot(df['population'], df['households'], hue=median_house_value)
sns.scatterplot(population_q, household_q, hue=median_house_value_q,palette=['red' for i in range(0,n)])
plt.show()

sns.scatterplot(df['total_rooms'], df['total_bedrooms'], hue=median_house_value)
sns.scatterplot(total_rooms_q, total_bedrooms_q, hue=median_house_value_q,palette=['red' for i in range(0,n)])
plt.show()

sns.scatterplot(df['latitude'], df['longitude'], hue=median_house_value)
sns.scatterplot(latitude_q, longitude_q, hue=median_house_value_q,palette=['red' for i in range(0,n)])
plt.show()
```

_N =4_

4 quantiles of the median House Value feature values are 119600(25%), 179700(50%), 264725(75%), 500001(100%).

![image](https://user-images.githubusercontent.com/77625823/141358809-fb736506-4957-4944-a0b8-6405d468f176.png)

Best purity score(with n=4): 0.327...

Best purity combination: [longitude, latitude], RobustScaler, GMM, k=7

![image](https://user-images.githubusercontent.com/77625823/141358831-7e8fc2c9-4061-4d14-914b-31fb897cfdf0.png)

