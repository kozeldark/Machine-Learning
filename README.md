# Machine-Learning

**ML Wisconsin breast cancer(PHW 1)**

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


