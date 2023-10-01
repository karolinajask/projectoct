import pandas as pd
import warnings
import sklearn                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")                                   # Warnings will appear only once

import matplotlib.pyplot as plt                                     # Popular plotting library

df = pd.read_csv("set_final.csv")

# df.info()

## converting categorical data into int

from sklearn.preprocessing import OneHotEncoder, StandardScaler
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['new_property','routing_code']
encoded_array = enc.fit_transform(df.loc[:,columns_to_one_hot])
df_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
df_sklearn_encoded = pd.concat([df,df_encoded],axis=1)

df_sklearn_encoded.drop(labels= columns_to_one_hot,axis=1,inplace=True)

df_sklearn_encoded.to_csv('encoded_set.csv')

# making data a numpy array like
x = df_sklearn_encoded.drop(['price_eur'], axis=1)
y = df_sklearn_encoded.price_eur
x = x.values
y = y.values



# print(df_sklearn_encoded.head())
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import bz2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)


# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler(),),
#                                         ('LR', LinearRegression())])))
# pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),
#                                            ('LASSO', Lasso())])))
# pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),
#                                         ('EN', ElasticNet())])))
# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),
#                                          ('KNN', KNeighborsRegressor())])))
# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),
#                                           ('CART', DecisionTreeRegressor())])))
# pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),
#                                          ('SVR', SVR())])))




# def modeling(models):
#     for name, model in models:
#         kfold = KFold(n_splits=10)
#         results = cross_val_score(model, rescaledx, y_train, cv = kfold, scoring='r2')
#         print(f'{name} = {results.mean()}')

# modeling(pipelines)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor

# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),
                                        ('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),
                                         ('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),
                                        ('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),
                                        ('ET', ExtraTreesRegressor())])))
kfold = KFold(n_splits=10)


for name, model in ensembles:
    cv_results = cross_val_score(model, rescaledx, y_train, cv=kfold, scoring='r2')
    print(f'{name} = {cv_results.mean()}')

