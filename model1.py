import pandas as pd
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")    
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle

df = pd.read_csv('set_final.csv')

## converting categorical data into int

from sklearn.preprocessing import OneHotEncoder, StandardScaler
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['new_property','routing_code']
encoded_array = enc.fit_transform(df.loc[:,columns_to_one_hot])
df_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
df_sklearn_encoded = pd.concat([df,df_encoded],axis=1)

df_sklearn_encoded.drop(labels= columns_to_one_hot,axis=1,inplace=True)

# making data a numpy array like
x = df_sklearn_encoded.drop(['price_eur'], axis=1)
y = df_sklearn_encoded.price_eur
x = x.values
y = y.values

# dividing data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)

# selecting and fitting the model for training
model = GradientBoostingRegressor()
model.fit(rescaledx, y_train)
# saving the trained mode
pickle.dump(model, open('gbm_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))


df_sklearn_encoded.to_csv('encoded_set.csv')