# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:58:51 2019

@author: raymond-cy.liu
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import re
import urllib.request as url
import os
import quandl
from time import mktime
import pandas_datareader.data as web
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
import collections
import time

from matplotlib import style
style.use('ggplot')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',500)

st = time.time()

def get_pe_ratio():
    response = url.urlopen('https://www.multpl.com/shiller-pe/table/by-month')
    pe_ratio_file = str(response.read())
    ended = False
    dates = []
    pe_ratios = []
    while not ended:
        try:
            date = re.search(r'<td class="left">(\w+\s\d+,\s\d+)</td>', pe_ratio_file)
            pe_ratio = re.search(r'<td class="right">(\d+\.\d+)', pe_ratio_file)
            date = date.group(1)
            pe_ratio = pe_ratio.group(1)
            pe_ratio_file = pe_ratio_file.replace(date,'').replace(pe_ratio,'')
#            FOR END OF MONTH DATE
            day = time.strptime(date, '%b %d, %Y')[2]
            unix_time = mktime(time.strptime(date, '%b %d, %Y')) - day * 86400
            date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
            dates.append(date)
            pe_ratios.append(float(pe_ratio))
#            time.sleep(1)
        except Exception as e:
            ended = True
            
    dictionary = {'Date': dates, 'PE_ratio': pe_ratios}
    df = pd.DataFrame(data = dictionary)
    df.set_index('Date').to_csv('PE_ratio.csv')    


def get_bonds():
    try:
        ten_yr = web.DataReader('DGS10','fred',start = '1970-01-30',end = datetime.today())
        one_yr = web.DataReader('DGS1','fred',start = '1970-01-30',end = datetime.today())
        bonds_yield = one_yr.join(ten_yr)
        bonds_yield.index = pd.to_datetime(bonds_yield.index)
        bonds_yield.rename(columns = {'DGS1': '1yr_yield', 'DGS10': '10yr_yield'}, inplace = True)
        bonds_yield.interpolate(method = 'linear', inplace = True)
#        bonds_yield.drop('1yr_yield', axis = 1, inplace = True)
        bonds_yield = bonds_yield.resample('1M').mean()
        bonds_yield.to_csv('bonds_yield.csv')
    except Exception as e:
        bonds_yield = pd.read_csv('bonds_yield.csv', index_col = 0)
    return bonds_yield


def get_interest_spreads():
    try:
        T10Y3M = web.DataReader('T10Y3M','fred',start = '1970-01-30',end = datetime.today())
        T10Y2Y = web.DataReader('T10Y2Y','fred',start = '1970-01-30',end = datetime.today())   
        T10YFF = web.DataReader('T10YFF','fred',start = '1970-01-30',end = datetime.today())
        interest_spread = T10Y3M.join([T10Y2Y, T10YFF])
        interest_spread.interpolate(method = 'linear', inplace = True)
        interest_spread = interest_spread.resample('1M').mean()
        interest_spread.to_csv('interest_spread.csv')
    except Exception as e:
        interest_spread = pd.read_csv('interest_spread.csv', index_col = 0)
    return interest_spread



def get_housing_affordability():
    try:
        price = web.DataReader('MSPNHSUS','fred',start = '1970-01-30',end = datetime.today())
        income = web.DataReader('A229RC0','fred',start = '1970-01-30',end = datetime.today())
        afford = income.join(price)
        afford['affordability'] = afford['MSPNHSUS'] / afford['A229RC0']
        afford.drop(['A229RC0', 'MSPNHSUS'], axis = 1, inplace = True)
        afford = afford.resample('1M').mean()
        afford.to_csv('housing_affordability.csv')
    except Exception as e:
        afford = pd.read_csv('housing_affordability.csv', index_col = 0)
    return afford



def get_non_fram():
    try:
        non_fram_df = web.DataReader('PAYEMS','fred',start = '1970-01-30',end = datetime.today())
        non_fram_df.index = pd.to_datetime(non_fram_df.index)
        non_fram_df = non_fram_df.resample('1M').mean()
        non_fram_df['Total Nonfarm pct'] = non_fram_df['PAYEMS'].shift(-1)
        non_fram_df['Total Nonfarm pct'] = (non_fram_df['Total Nonfarm pct'] - non_fram_df['PAYEMS']) / non_fram_df['PAYEMS'] * 100
#        non_fram_df.drop('PAYEMS', axis = 1, inplace = True)
        non_fram_df.to_csv('Total Nonfarm.csv')
    except Exception as e:
        non_fram_df = pd.read_csv('Total Nonfarm.csv', index_col = 0)
    return non_fram_df



def get_industrial_production():
    try:
        industrial_production = web.DataReader('INDPRO','fred',start = '1970-01-30',end = datetime.today())
        industrial_production.index = pd.to_datetime(industrial_production.index)
        industrial_production = industrial_production.resample('1M').mean()
        industrial_production.to_csv('Industrial production.csv')
    except Exception as e:
        industrial_production = pd.read_csv('Industrial production.csv', index_col = 0)
    return industrial_production

 

def get_unemployment():
    try:
        unemployment = web.DataReader('UNRATE','fred',start = '1970-01-30',end = datetime.today())
        unemployment.index = pd.to_datetime(unemployment.index)
        unemployment = unemployment.resample('1M').mean()
        unemployment.to_csv('Unemployment.csv')
    except Exception as e:
        unemployment = pd.read_csv('Unemployment.csv', index_col = 0)
    return unemployment

    

def get_credit_risk():
    try:
        credit_risk = web.DataReader('NFCICREDIT','fred',start = '1970-01-30',end = datetime.today())
        credit_risk.index = pd.to_datetime(credit_risk.index)
        credit_risk = credit_risk.resample('1M').mean()
        credit_risk.to_csv('Credit_risk.csv')
    except Exception as e:
        credit_risk = pd.read_csv('Credit_risk.csv', index_col = 0)
    return credit_risk



def get_effective_fed_fund_rate():
    try:
        fed_fund_rate = web.DataReader('FF','fred',start = '1970-01-30',end = datetime.today())
        fed_fund_rate.index = pd.to_datetime(fed_fund_rate.index)
        fed_fund_rate = fed_fund_rate.resample('1M').mean()
        fed_fund_rate['Fed_fund_pct'] = fed_fund_rate['FF'].shift(-1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        fed_fund_rate['Fed_fund_pct'] = (fed_fund_rate['Fed_fund_pct'] - fed_fund_rate['FF']) / fed_fund_rate['FF'] * 100
#        fed_fund_rate.drop('FEDFUNDS', axis = 1, inplace = True)
        fed_fund_rate.to_csv('Effective Federal Funds Rate.csv')
    except Exception as e:
        fed_fund_rate = pd.read_csv('Effective Federal Funds Rate.csv', index_col = 0)
    return fed_fund_rate


def get_heavy_truck():
    try:
        heavy_truck = web.DataReader('HTRUCKSSAAR','fred',start = '1970-01-30',end = datetime.today())
        heavy_truck.index = pd.to_datetime(heavy_truck.index)
        heavy_truck = heavy_truck.resample('1M').mean()
        heavy_truck['Heavy_truck_pct'] = heavy_truck['HTRUCKSSAAR'].shift(-1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        heavy_truck['Heavy_truck_pct'] = (heavy_truck['Heavy_truck_pct'] - heavy_truck['HTRUCKSSAAR']) / heavy_truck['HTRUCKSSAAR'] * 100
#        fed_fund_rate.drop('FEDFUNDS', axis = 1, inplace = True)
        heavy_truck.to_csv('Heavy Truck.csv')
    except Exception as e:
        heavy_truck = pd.read_csv('Heavy Truck.csv', index_col = 0)
    return heavy_truck
   
     
def real_gdp():
    try:
        gdp_1yr_pct = web.DataReader('A191RO1Q156NBEA','fred',start = '1970-01-30',end = datetime.today())
        gdp_1yr_pct.index = pd.to_datetime(gdp_1yr_pct.index)
        gdp_1yr_pct = gdp_1yr_pct.resample('1M').mean()
        gdp_1yr_pct = gdp_1yr_pct.interpolate(method ='linear', limit_direction ='forward')         
        gdp_1yr_pct.to_csv('Real GDP 1yr comparison.csv')
    except Exception as e:
        gdp_1yr_pct = pd.read_csv('Real GDP 1yr comparison.csv', index_col = 0)
    return gdp_1yr_pct
    

def get_rsi():
    shift = 14
    sp500 = pd.read_csv('^GSPC.csv', index_col = 0)
    sp500 = sp500[['Adj Close']]
    sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500.resample('1M').mean()
    sp500['pct'] = sp500['Adj Close'].shift(1)
    sp500['pct'] = (-sp500['pct'] + sp500['Adj Close'])# / sp500['Adj Close'] 
    sp500['gain'] = sp500['pct']
    sp500['loss'] = sp500['pct']
    sp500['gain'].loc[sp500['pct'] < 0] = 0
    sp500['loss'].loc[sp500['pct'] >= 0] = 0
    sp500['avg gain'] = np.nan
    sp500['avg loss'] = np.nan
    sp500.iloc[shift, 4] = sp500.iloc[:shift+1,2].mean()
    sp500.iloc[shift, 5] = sp500.iloc[:shift+1,3].mean()
    for i in range(shift+1, len(sp500)):
        sp500.iloc[i, 4] = (sp500.iloc[i-1, 4] * (shift-1) + sp500.iloc[i, 2]) / shift
        sp500.iloc[i, 5] = (sp500.iloc[i-1, 5] * (shift-1) + sp500.iloc[i, 3]) / shift
    sp500['avg loss'] = sp500['avg loss'].abs()
    sp500['RSI'] = 100 - (100 / (1 + sp500['avg gain'] / sp500['avg loss']))
    sp500 = sp500['RSI']
    sp500.to_csv('RSI.csv')
    sp500 = sp500.loc['1970-01-30':]
    return sp500
	

def get_sp500(shift = 24, buy_pct = 10, sell_pct = -10):
    sp500 = web.DataReader('^GSPC','yahoo', start = '1970-01-30',end = datetime.today())
    sp500.to_csv('^GSPC.csv')
    sp500_price = sp500.drop('Volume', axis = 1).resample('1M').mean()
    sp500_volume = sp500['Volume'].resample('1M').sum()
    sp500 = sp500_price.join(sp500_volume)
    sp500['Volatility'] = (sp500['High'] - sp500['Low']) / sp500['Low'] * 100
    sp500['pct_change'] = (sp500['Adj Close'].shift(-1) - sp500['Adj Close']) / sp500['Adj Close'] * 100
    sp500['future'] = sp500['Adj Close'].shift(-shift)
    sp500['diff'] = (sp500['future'] - sp500['Adj Close']) / sp500['Adj Close'] * 100
    sp500['label'] = np.nan
    sp500['label'].loc[sp500['diff'] >= buy_pct] = 1
    sp500['label'].loc[sp500['diff'] < sell_pct] = -1
    sp500.label.iloc[:-shift].fillna(0, inplace = True)
    drop_list = ['High', 'Low', 'Open', 'Close',  'future',  'diff']
    sp500.drop(drop_list, axis = 1, inplace = True)
    to_save = []
    for item in sp500.columns.values:
        if item not in drop_list:
            to_save.append(item)
    for item in to_save:
        sp500.rename(columns = {item: 'sp500_' + item}, inplace = True)
    sp500.to_csv('S&P500.csv')



def build_dataset(shift, test, grab_data, st_date):
	
	if grab_data:
		combined_df = pd.DataFrame()
		try:
			get_pe_ratio()
			get_sp500(buy_pct = 15, sell_pct = -15) 
		except:
			pass
		sp500 = pd.read_csv('S&P500.csv', index_col = 0)
		pe_ratio = pd.read_csv('PE_ratio.csv', index_col = 0)
		bonds_yield = get_bonds()
		interest_spread = get_interest_spreads()
		housing_affordability = get_housing_affordability()
		unemployment = get_unemployment()
		industrial_production = get_industrial_production()
		non_fram = get_non_fram()
		credit_risk = get_credit_risk()
		fed_fund_rate = get_effective_fed_fund_rate()
		heavy_truck = get_heavy_truck()
		gdp_1yr_pct = real_gdp()
		rsi = get_rsi()
		
		dfs = [pe_ratio,
	           bonds_yield, 
	           interest_spread,
	           fed_fund_rate,
	           housing_affordability,
	           non_fram,
	           industrial_production,
	           gdp_1yr_pct,
	           credit_risk,
	           heavy_truck,
	           unemployment,
	           sp500,
	           rsi,
			   ]
		
		if combined_df.empty:
		   combined_df = dfs[0]
		for df in dfs[1:]:
			combined_df = combined_df.copy().join(df, how = 'outer')
		combined_df.fillna(0, inplace = True)
	else:
		combined_df = pd.read_csv('combined_df.csv', index_col = 0)
	combined_df.index = pd.to_datetime(combined_df.index)
	
	if st_date == None:
		st_index = combined_df.index.get_loc(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d'), method = 'pad')
	else:
		st_index = combined_df.index.get_loc(st_date, method = 'pad')
	
# 	Manually scale data till st_index
	scaled_df = combined_df.sort_index().drop('sp500_label', axis = 1).iloc[:st_index + 1]
	scaled_df = (scaled_df - scaled_df.mean()) / scaled_df.std()
	scaled_df['sp500_label'] = combined_df['sp500_label'].iloc[:st_index + 1]
# 	print(scaled_df.tail(15))
	
	clf_df = scaled_df.sort_index().iloc[:-shift]
	pred_df = scaled_df.sort_index().iloc[-shift:]
	clf_length = len(clf_df) 
	test_size = int(clf_length * test)
	clf_df = clf_df.copy().reindex(np.random.permutation(clf_df.index))
	X_train = np.array(clf_df.iloc[:-test_size].drop('sp500_label', axis = 1))
	X_test = np.array(clf_df.iloc[-test_size:].drop('sp500_label', axis = 1))
	y_train = np.array(clf_df['sp500_label'].iloc[:-test_size])
	y_test = np.array(clf_df['sp500_label'].iloc[-test_size:])
	pred = np.array(pred_df.drop('sp500_label', axis = 1))

	if grab_data:
	    combined_df.to_csv('combined_df.csv')
		
	return X_train, y_train, X_test, y_test, pred, clf_length, test_size, clf_df.index, pred_df.index


def Analysis(grab_data, st_date, shift = 6, test = 0.2):
	
    X_train, y_train, X_test, y_test, pred, clf_length, test_size, dates, future_dates = build_dataset(shift, test, grab_data, st_date)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, clf_length, test_size, dates)
    clf = svm.SVC()
    # clf = RandomForestClassifier() 
    clf.fit(X_train, y_train)
    correct = 0
    correct_pos, correct_0, correct_neg = 0, 0, 0
    result = {}
    unique, counts = np.unique(y_test, return_counts = True)
    test_result = dict(zip(unique, counts))
    future_result = {}
	
    for item in range(1, test_size + 1):
        # print(dates[-item].strftime('%Y-%m-%d'), clf.predict(X_test[[-item]])[0], X_test[-item], y_test[-item])
        result[dates[-item].strftime('%Y-%m-%d')] = clf.predict(X_test[[-item]])[0]
        if clf.predict(X_test[[-item]])[0] == y_test[-item]:
            correct += 1
            if y_test[-item] == 1:
                correct_pos += 1
            elif y_test[-item] == 0:
                correct_0 += 1
            else:
                correct_neg += 1
    
    for item in range(1, len(pred) + 1):
        # print(dates[-item].strftime('%Y-%m-%d'), clf.predict(X_test[[-item]])[0], y_test[[-item]][0])
        future_result[future_dates[-item].strftime('%Y-%m-%d')] = clf.predict(pred[[-item]])[0]
    return result, clf_length, future_result, correct / (test_size + 1) * 100, \
		correct_pos / test_result[1] * 100, correct_0 / test_result[0] * 100, correct_neg / test_result[-1] * 100


def plot_acc_PL(run_num = 10, plot = False, grab_data = False, st_date = None):
	test_result, future_pred, acc = [], [], []
	acc_pos, acc_0, acc_neg = [], [], []
	for i in range(run_num):
		if i == 0 and grab_data:
			a, _, b, c, d, e, f = Analysis(grab_data = True, st_date = st_date)
		else:
			a, _, b, c, d, e, f = Analysis(grab_data = False, st_date = st_date)
		test_result.append(a)
		future_pred.append(b)
		acc.append(c)
		acc_pos.append(d)
		acc_0.append(e)
		acc_neg.append(f)
	df = pd.DataFrame(test_result).T
	future_df = pd.DataFrame(future_pred).T
	actual_df = pd.read_csv('combined_df.csv', index_col = 0)
	if st_date == None:
		st_index = actual_df.index.get_loc(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d'), method = 'pad')
	else:
		st_index = actual_df.index.get_loc(st_date, method = 'pad')
		
	for date in df.index:
		df.loc[date, 'Past Pred'] = df.loc[date].mean()
		df.loc[date, 'Actual'] = actual_df.loc[date, 'sp500_label']  
		df.loc[date, 'SP500_pct'] = actual_df.loc[date, 'sp500_pct_change']
	for date in future_df.index:
		future_df.loc[date, 'Future Pred'] = future_df.loc[date].mean()
# =============================================================================
# 	print(future_df, '\n')
# 	print(f'Training till: {actual_df.index[st_index] if st_date != None else actual_df.index[-1]}')
# 	print(f'Buy Accuracy: {round(sum(acc_pos) / len(acc_pos), 2)}%')
# 	print(f'Hold Accuracy: {round(sum(acc_0) / len(acc_0), 2)}%')
# 	print(f'Sell Accuracy: {round(sum(acc_neg) / len(acc_neg), 2)}%')
# 	print(f'Overall Accuracy: {round(sum(acc) / len(acc_pos), 2)}%')
# =============================================================================
	df.index = pd.to_datetime(df.index)
	if plot:
		plt.scatter(df.index, df['Past Pred'], s = 1, color = 'red')
		plt.scatter(future_df.index, future_df['Future Pred'], s = 10, color = 'blue')
		plt.plot(df.index, df['Actual'], color = 'black', alpha = 0.3)
		plt.show()
		

for i in range(2005, 2020):
	for j in range(1, 13, 3):
		st_date = f'{i}-0{j}-15' if j < 10 else f'{i}-{j}-15'
		print(st_date)
		plot_acc_PL(plot = True, grab_data = False, st_date = st_date, run_num = 50)
		
print(f'Time elapsed: {int(time.time() - st)} secs')

