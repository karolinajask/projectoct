import pandas as pd
import warnings
import sklearn                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")    
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle

df2 = pd.read_csv('set_lessf.csv')

## converting categorical data into int
## https://medium.com/@sushmit86/one-hot-encoding-sklearn-vs-pandas-de32947ef4ef 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
columns_to_one_hot = ['new_property','routing_code']
encoded_array = enc.fit_transform(df2.loc[:,columns_to_one_hot])
df_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
df_sklearn_encoded = pd.concat([df2,df_encoded],axis=1)
pickle.dump(enc, open('enc.pkl', 'wb'))

df_sklearn_encoded.drop(labels= columns_to_one_hot,axis=1,inplace=True)

##https://blog.finxter.com/how-i-built-a-house-price-prediction-app-using-streamlit/
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
model = RandomForestRegressor()
model.fit(rescaledx, y_train)
# saving the trained mode
pickle.dump(model, open('gbm_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))

df_sklearn_encoded.to_csv('encoded_setlessf.csv')