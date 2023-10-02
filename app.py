import streamlit as st
import pickle
import pandas as pd
import warnings     
import sklearn
warnings.filterwarnings("ignore") 

def main():

    ##mapping codes

    fdict = {'A42':['Garristown'],
             'A45':['Oldtown'],            
             'A94':['Blackrock', 'Booterstown', 'Mount Merrion','Stillorgan','Monkstown'],
             'A96':['Dalkey','Killiney','Dunleary', 'Dun Laoghaire', 'Sandycove','Glenageary'],
             'D01':['Dublin1', 'Dublin 1'],
             'D02':['Dublin2', 'Dublin 2'],
             'D03':['Dublin3', 'Dublin 3','Clontarf','Fairview'],
             'D04':['Dublin4', 'Dublin 4','Ringsend','Sandymount','Donnybrook','Ballsbridge'],
             'D05':['Dublin5', 'Dublin 5','Raheny','Killester','Artane','Coolock','Harmonstown'],
             'D06':['Dublin6', 'Dublin 6','Ranelagh','Rathgar','Rathmines','Milltown'],
             'D6W':['Dublin6W', 'Dublin 6W', 'Templeogue','Terenure','Kimmage','Harold\'s Cross'],
             'D07':['Dublin7', 'Dublin 7', 'Phibsborough', 'Smithfield','Cabra','Stoneybatter'],
             'D08':['Dublin8', 'Dublin 8','Portobello','Islandbridge','Kilmainham','Dolphin\'s Barn'],
             'D09':['Dublin9', 'Dublin 9','Beaumont', 'Donnycarney', 'Drumcondra'],
             'D10':['Dublin10', 'Dublin 10','Inchicore','Ballyfermot'],
             'D11':['Dublin11', 'Dublin 11','Finglas','Ballymun'],
             'D12':['Dublin12', 'Dublin 12','Crumlin','Drimnagh','Walkinstown'],
             'D13':['Dublin13', 'Dublin 13', 'Baldoyle', 'Bayside', 'Donaghmede', 'Clongriffin', 'Sutton', 'Howth'],
             'D14':['Dublin14', 'Dublin 14','Churchtown', 'Clonskeagh', 'Dundrum', 'Goatstown', 'Rathfarnham', 'Windy Arbour'],
             'D15':['Dublin15', 'Dublin 15','Ashtown', 'Blanchardstown', 'Castleknock', 'Coolmine', 'Clonsilla', 'Corduff', 'Mulhuddart', 'Tyrrelstown','Ongar'],
             'D16':['Dublin16', 'Dublin 16','Knocklyon','Balinteer'],
             'D17':['Dublin17', 'Dublin 17','Balgriffin','Darndale'],
             'D18':['Dublin18', 'Dublin 18','Carrickmines','Stepaside','Sandyford','Shankill'],
             'D20':['Dublin20', 'Dublin 20','Chapelizod', 'Palmerstown'],
             'D22':['Dublin22', 'Dublin 22','Clondalkin','Newcastle'],
             'D24':['Dublin24', 'Dublin 24','Firhouse', 'Jobstown', 'Old Bawn', 'Tallaght'],
             'K32':['Balbriggan','Naul'],
             'K34':['Skerries'],
             'K36':['Malahide','Donabate'],
             'K45':['Lusk'],
             'K56':['Rush'],
             'K67':['Swords'],
             'K78':['Lucan']
        }

    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'dublin>HOUSEPRICE DUBLIN</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)

    ### https://discuss.streamlit.io/t/change-input-text-font-size/29959/7
    tabs_font_css = """
    <style>
    div[class*="stTextArea"] label p {
    font-size: 26px;
    color: red;
    }

    div[class*="stNumberInput"] label p {
    font-size: 26px;   
    }

    div[class*="stSelect"] label p {
    font-size: 26px;    
    }

    div[class*="stRadio"] label p {
    font-size: 26px;    
    }
    </style>
    """

    st.write(tabs_font_css, unsafe_allow_html=True)

    
    year = st.number_input('Enter the year you want to buy / sell',help='Please choose a year between 2024 and 2050', step =1, value=2024, min_value=2024, max_value=2050)
    routing_code = st.selectbox('Select routing code', ('A42','A94','A96','D01','D02','D03','D04','D05','D06','D6W','D07','D08','D09','D10','D11','D12','D13','D13','D15','D16','D18','D22','D24',
                                                               'K32','K34','K36','K45','K67','K78'),help='Routing code consists of first 3 characters of your Eircode,\n for example: D01 ABC -> routing code is D01.')
    st.write('Can\'t find your routing code?. Let us know and we\'ll fix it for you ! Contact our team at support@hpdublin.ie or call +353 55555 🚨')
    new_property = st.selectbox('New or second-hand?',('New Dwelling house /Apartment', 'Second-Hand Dwelling house /Apartment'))
    button = st.button('Predict')
    # if button is pressed
    if button:
        # make prediction
        result = predict(year, routing_code,new_property)
        st.success(f'The value of the house is EUR {result}')
    with st.container():
        st.write("Not sure about your routing code?")
        a = st.text_input('Enter the area/district below')
     
        button = st.button('Check  my routing code')
         # if button is pressed
        if button:
            for key, value in fdict.items():
                if a in value:
                    st.write('Your routing code is',key)
 
            st.write('Want to add your location in our database? Contact our team at support@hpdublin.ie or call +353 55555 🚨')   

    #print(fdict)

# load the train model
with open('gbm_model.pkl', 'rb') as gbm:
    model = pickle.load(gbm)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)

# load the Encoder
with open('enc.pkl', 'rb') as en:
    enc = pickle.load(en)

def predict(year,routing_code,new_property): 
    # processing user input
    c = {'year':[year],'new_property':[new_property],'routing_code':[routing_code]}
    df= pd.DataFrame(data=c)
    # encoding the data
    z = enc.transform(df[["new_property","routing_code"]]) 
    #print(z)
    dfx = pd.DataFrame(z,columns=enc.get_feature_names_out())
    dfv = pd.concat([df,dfx],axis=1)
    dfv.drop(labels= ["new_property","routing_code"],axis=1,inplace=True)
    #dfv.to_csv('v.csv')
    #scaling the data
    scaler.transform(dfv)
    # making predictions
    prediction = model.predict(dfv)
    result = int(prediction)
    return result

if __name__ == '__main__':
    try:
        main()
    except :
        st.error('We are sorry, something went wrong on our end. Please try again later or contact our support team at support@hpdublin.ie or call +353 55555 ', icon="🚨")
