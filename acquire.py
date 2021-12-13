import pandas as pd
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from scipy.stats import pearsonr, spearmanr
from scipy import stats
# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing

# import imputing function
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

'''
1. Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. Obtain your data from the Codeup Data Science Database.
'''

def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df

def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to a csv file if a local file does not exist, and returns a df.
    ''' 
    if os.path.isfile('titanic.csv'):
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic.csv', index_col=0)
        
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv('titanic.csv')

        # Return the dataframe to the calling code
    return df


'''
2. Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. The returned data frame should include the actual name of the species in addition to the species_ids. Obtain your data from the Codeup Data Science Database.
'''
def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT *
                FROM measurements
                JOIN species USING(species_id);
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df


def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df


'''
3. Make a function named get_telco_data that returns the data from the telco_churn database in SQL. In your SQL, be sure to join all 4 tables together, so that the resulting dataframe contains all the contract, payment, and internet service options. Obtain your data from the Codeup Data Science Database.
'''

def new_telco_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        return df
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df

#******************************************** Acquire and Wrangle Zillow Data **************************************************#
# This function connects to the Codeup database.
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
# This function fetches the specified data and stores it as a Pandas dataframe
def zillow_sql():
    '''
    This function reads the zillow data from the Codeup database into a Pandas dataframe.
    '''
    sql_query = """
                SELECT * FROM predictions_2017
                left join properties_2017 using(parcelid)
                where properties_2017.propertylandusetypeid = 261;
                """
    
    # Read in DataFrame from Codeup db.
    zillow_df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return zillow_df

# This function caches the zillow data.

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        zillow_df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        zillow_df = zillow_sql()
        
        # Cache data
        zillow_df.to_csv('zillow.csv')
        
    return zillow_df

# This function removes outliers.
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

# This function
def wrangle_zillow():
    # Store zillow data into a variable
    zillow_df = get_zillow_data()
    
    # Use these features for the 'MVP' model.
    orig_draft_features = ['landtaxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt', 'fips']
    orig_draft_features = zillow_df[orig_draft_features]
    
    # Renaming columns to make referencing easier.
    orig_draft_features = orig_draft_features.rename(columns = {'landtaxvaluedollarcnt':'tax_value','calculatedfinishedsquarefeet':'sqft','bedroomcnt':'bedrooms','bathroomcnt':'bathrooms'})
    
    # Run the 'remove_outliers' function and update dataframe.
    orig_draft_features = remove_outliers(orig_draft_features, 1.5, ['tax_value', 'sqft', 'bedrooms', 'bathrooms'])

    train_validate, test = train_test_split(orig_draft_features, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test
    
def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    
    # new column names
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    # Fit the scaler on the train
    scaler.fit(train[columns_to_scale])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test
    
def select_kbest(X, y, k):
    # make the object
    kbest = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression, k=k)

    # fit the object
    kbest.fit(X, y)
    
    # use the object (.get_support() is that array of booleans to filter the list of column names)
    return X.columns[kbest.get_support()].tolist()

def model_evaluation(X_train_scaled, X_validate_scaled, y_train, y_validate):    
    
    rmse_train = mean_squared_error(y_train.tax_value,
                                y_train.tax_value_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_mean) ** (0.5) 
    metric_df = pd.DataFrame(data=[
            {
                'model': 'mean_baseline', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    # create the model object(thing)
    lm = LinearRegression()
    # fit the thing
    lm.fit(X_train_scaled, y_train.tax_value)
    # predict train
    # use the thing!
    y_train['tax_value_pred_lm'] = lm.predict(X_train_scaled)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm) ** (1/2)
    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(X_validate_scaled)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm) ** (1/2)
    metric_df = metric_df.append({
    'model': 'OLS Regressor', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    
    # create the LassoLars model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** (1/2)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars) ** (1/2)

    metric_df = metric_df.append({
    'model': 'lasso_alpha0.1', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)


    # fit the model to our training data. 
    glm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm) ** (1/2)

    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm) ** (1/2)
    
    metric_df = metric_df.append({
    'model': 'glm_poisson', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    
    return metric_df
