import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.model_selection import train_test_split



def split_data(data):
    X, y =  data.iloc[:,1:].values, data['Settlement Point Price'].values
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=42)


    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
    return X_train, y_train, X_test, y_test



def import_df():
    load = pd.read_excel('data/Native_Load_2022.xlsx')
    weather = pd.read_csv('data/caiso_2021-22.csv')
    actuals = pd.read_excel('data/ErcotDamActuals.xlsx', sheet_name=None)
    actuals = pd.concat(actuals)


    ## Clean
    weather.DateTime = pd.to_datetime(weather.DateTime)
    weather.set_index('DateTime', inplace=True)
    weather = weather.resample('1H').mean()
    weather = weather.truncate(before='2022-02-01 00:00:00')
    weather = weather.loc[:,['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural Gas', 'Large Hydro', 'Batteries', 'Imports']]
    weather[weather < 0] = 0 #replace negatives by 0s

    dic = {}
    west_actuals = actuals.loc[actuals['Settlement Point']=='HB_WEST',:]
    # print(west_actuals)
    for i in list(set(west_actuals['Hour Ending'])):
        dic[i] = str(int(i[0:2])-1)+i[2:]
    west_actuals=west_actuals.replace({"Hour Ending": dic})
    west_actuals.loc[:,'time'] = pd.to_datetime(west_actuals['Delivery Date'] + ' ' + west_actuals['Hour Ending']).values
    west_actuals.set_index('time', inplace=True)
    west_actuals = west_actuals.loc[:,['Settlement Point Price']]

    load[['A', 'B']] = load['Hour Ending'].str.split(' ', 1, expand=True)
    load.dropna(inplace=True)
    load=load.replace({"B": dic})
    load.loc[:,'time'] = pd.to_datetime(load['A'] + ' ' + load['B']).values
    load.set_index('time', inplace=True)
    load = load.drop(columns=['A', 'B', 'Hour Ending'])

    df = west_actuals.join(weather, how='inner')
    df = df.join(load, how='inner')
    df['Settlement Point Price'] = df['Settlement Point Price'].shift(24)
    df.dropna(inplace=True)
    return df
