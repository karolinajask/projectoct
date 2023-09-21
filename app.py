import streamlit as st
import pickle
import pandas as pd

def main():
    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'dublin >House Price Prediction App</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    year = left.number_input('Enter the year you want to buy', step =1, value=2024, min_value=2024, max_value=2040)
    routing_code = st.selectbox('Please select routing code', ('D01', 'A94'))
    new_property = st.selectbox('New or 2nd hand?', ('New', 'Second-hand'))
    button = st.button('Predict')
    # if button is pressed
    if button:
        # make prediction
        result = predict(year, routing_code,new_property)
        st.success(f'The value of the house is ${result}')

# load the train model
with open('gbm_model.pkl', 'rb') as gbm:
    model = pickle.load(gbm)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)

def predict(year,routing_code,new_property): #### KJ 21/09/2023 - Not sure what needs to be done here so categorical variables routing_code,new_property are processed by the model so prediction is returned
    # processing user input
    ###
    lists = [year, routing_code,new_property]
    df = pd.DataFrame(lists).transpose()
    # scaling the data
    scaler.transform(df)
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result



if __name__ == '__main__':
    main()

