#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import env
import pydataset as data
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire
# Create function to split Telco dataset

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test


# In[2]:


# Create function that cleans up the Telco dataset

def prep_telco(df):
    telco_df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id','customer_id'])
    
    # drop null values stored as whitespace
    telco_df['total_charges'] = telco_df['total_charges'].str.strip()
    telco_df = telco_df[telco_df.total_charges != '']
    
    # convert total_charges into a float
    telco_df['total_charges'] = telco_df['total_charges'].astype(float)
    
    # convert binary categorical variables to numeric
    telco_df['gender_encoded'] = telco_df.gender.map({'Female': 1, 'Male': 0})
    telco_df['partner_encoded'] = telco_df.partner.map({'Yes': 1, 'No': 0})
    telco_df['dependents_encoded'] = telco_df.dependents.map({'Yes': 1, 'No': 0})
    telco_df['phone_service_encoded'] = telco_df.phone_service.map({'Yes': 1, 'No': 0})
    telco_df['paperless_billing_encoded'] = telco_df.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_df['churn_encoded'] = telco_df.churn.map({'Yes': 1, 'No': 0})
    

    dummy_df = pd.get_dummies(telco_df[['gender','partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','contract_type','internet_service_type','payment_type']],dummy_na=False)
    
    clean_telco_df = pd.concat([telco_df, dummy_df], axis=1)
    
    train, validate, test = split_telco_data(clean_telco_df)
    
    return train, validate, test

    '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


    

