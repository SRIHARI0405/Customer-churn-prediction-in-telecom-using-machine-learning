

import pandas as pd
import streamlit as st
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
st.title('Telecom customer Churn prediction')
st.header('User Input Parameters')
def user_input_features():
    gender = st.selectbox('gender',('Female','Male'))
    SeniorCitizen = st.selectbox('SeniorCitizen',('Yes','No'))
    Partner= st.selectbox('Partner',('Yes','No'))
    Dependents= st.selectbox('Dependents',('Yes','No'))
    MultipleLines= st.selectbox('MultipleLines',('Yes','No','No phone service'))
    InternetService = st.selectbox('InternetService',('DSL','Fiber optic','No'))
    OnlineSecurity= st.selectbox('OnlineSecurity', ('Yes', 'No', 'No internet service'))
    OnlineBackup= st.selectbox('OnlineBackup',('Yes', 'No', 'No internet service'))
    DeviceProtection= st.selectbox('DeviceProtection',('Yes', 'No', 'No internet service'))
    TechSupport= st.selectbox('TechSupport',('Yes', 'No', 'No internet service'))
    StreamingTV= st.selectbox('StreamingTV',('Yes', 'No', 'No internet service'))
    StreamingMovies= st.selectbox('StreamingMovies',('Yes', 'No', 'No internet service'))
    tenure= st.number_input('tenure', 0, 1680)
    PhoneService= st.selectbox('PhoneService',('Yes','No'))
    Contract= st.selectbox('Contract',('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling= st.selectbox('PaperlessBilling',('Yes','No'))
    PaymentMethod= st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges= st.number_input('MonthlyCharges')
    TotalCharges= st.number_input('TotalCharges')
    data = {'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents':  Dependents,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection':DeviceProtection,
            'TechSupport':TechSupport,
            'StreamingTV':StreamingTV,
            'StreamingMovies':StreamingMovies,
            'tenure':tenure,
            'PhoneService':PhoneService,
            'Contract':Contract,
            'PaperlessBilling':PaperlessBilling,
            'PaymentMethod':PaymentMethod,
            'MonthlyCharges':MonthlyCharges,
            'TotalCharges':TotalCharges }
    features = pd.DataFrame(data,index=[0])
    return features   

df = user_input_features()
df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)
yes_no_columns = ['SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
for col in yes_no_columns:
    df[col].replace({'Yes': 1,'No': 0},inplace=True)

df['gender'].replace({'Female':1,'Male':0},inplace=True)
if df['InternetService'][0]=='DSL':
    df['InternetService_DSL']=1
    df['InternetService_Fiber optic']=0
    df['InternetService_No']=0
    df.drop('InternetService',axis='columns')
    
if df['InternetService'][0]=='Fiber optic':
    df['InternetService_DSL']=0
    df['InternetService_Fiber optic']=1
    df['InternetService_No']=0
    df.drop('InternetService',axis='columns')
    
if df['InternetService'][0]=='No':
    df['InternetService_DSL']=0
    df['InternetService_Fiber optic']=0
    df['InternetService_No']=1
    df.drop('InternetService',axis='columns')

del df['InternetService']
if df['Contract'][0]=='Month-to-month':
    
    df['Contract_Month-to-month']=1
    df['Contract_One year']=0
    df['Contract_Two year']=0
    df.drop('Contract',axis='columns')
    
if df['Contract'][0]=='One year':
    
    df['Contract_Month-to-month']=0
    df['Contract_One year']=1
    df['Contract_Two year']=0
    df.drop('Contract',axis='columns')
if df['Contract'][0]=='Two year':
    
    df['Contract_Month-to-month']=0
    df['Contract_One year']=0
    df['Contract_Two year']=1
    df.drop('Contract',axis='columns')
    
del df['Contract']
if df['PaymentMethod'][0]=='Electronic check':
    df['PaymentMethod_Electronic check']=1
    df['PaymentMethod_Mailed check']=0
    df['PaymentMethod_Bank transfer (automatic)']=0
    df['PaymentMethod_Credit card (automatic)']=0
    df.drop('PaymentMethod',axis=1)
if df['PaymentMethod'][0]=='Mailed check':
    df['PaymentMethod_Electronic check']=0
    df['PaymentMethod_Mailed check']=1
    df['PaymentMethod_Bank transfer (automatic)']=0
    df['PaymentMethod_Credit card (automatic)']=0
    df.drop('PaymentMethod',axis=1)
if  df['PaymentMethod'][0]=='Bank transfer (automatic)':
    df['PaymentMethod_Electronic check']=0
    df['PaymentMethod_Mailed check']=0
    df['PaymentMethod_Bank transfer (automatic)']=1
    df['PaymentMethod_Credit card (automatic)']=0
    df.drop('PaymentMethod',axis=1)  
if df['PaymentMethod'][0]=='Credit card (automatic)':
    df['PaymentMethod_Electronic check']=0
    df['PaymentMethod_Mailed check']=0
    df['PaymentMethod_Bank transfer (automatic)']=0
    df['PaymentMethod_Credit card (automatic)']=1
    df.drop('PaymentMethod',axis=1) 

del df['PaymentMethod']

df2=df

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df_churn=pd.read_csv('churn_data.csv')
df_cust=pd.read_csv('customer_data.csv')
df_internet=pd.read_csv('internet_data.csv')
data= (df_cust.merge(df_internet, on =['customerID'], suffixes=('_left','_right') )).merge(df_churn, on=['customerID'], suffixes=('_left','_right') )
data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
data.drop(labels=data[data['tenure'] == 0].index, axis=0, inplace=True)
data=data.drop(['customerID'],axis=1)
data.replace('No internet service','No',inplace=True)
data.replace('No phone service','No',inplace=True)
yes_no_columns_1 = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns_1:
    data[col].replace({'Yes': 1,'No': 0},inplace=True)
data['gender'].replace({'Female':1,'Male':0},inplace=True)
data2 = pd.get_dummies(data=data, columns=['InternetService','Contract','PaymentMethod'])
data2[cols_to_scale] = scaler.fit_transform(data2[cols_to_scale])
X = data2.drop(columns = ['Churn'])
y = data2['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)
from imblearn.over_sampling import SMOTE

smote=SMOTE()
x_rec,y_rec=smote.fit_resample(X,y)
print(x_rec,y_rec)
XGB=xgb.XGBClassifier()
XGB_Model = XGB.fit(x_rec,y_rec)
print(XGB_Model)
Output = XGB_Model.predict(X_train)
if Output[0]==1:
    st.subheader('Customer is likely to Churn')
else:
    st.subheader('Customer is unlikely to Churn')

