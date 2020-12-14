#Import the python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#Get the stock symbols in the portfolio
#FAANG
assets = ['FB', 'AMZN','AAPL','NFLX','GOOG','IBM']
#, 'MSFT', 'MCD','MA','TSLA','KO','NVDA','HD','ORCL','SAP','CSCO','VMW','INTC','NOW','BLK','EBAY','DXC','VMC','VLO','WM','WST']

#Assign weights to the stocks.

weights = np.random.random(len(assets))
weights = weights/np.sum(weights)

#Get the stocks/portfolio starting date
stockStartDate = '2018-01-01'

#Get the stocks ending date (today)
today = datetime.today().strftime('%Y-%m-%d')

#Create a dataframe to store adjusted close price of the stocks
df = pd.DataFrame()

#Store the adjusted close price into the df
for stock in assets :
  df[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end = today)['Adj Close']


# Visually show the stock / portfolio
plt.style.use('seaborn-white')
df.plot(legend=0, figsize=(10,6), grid=True, title='Daily Returns of the ETF Stocks')
plt.tight_layout()
plt.ylabel('Adj. Price USD ($)', fontsize = 10)
plt.legend(df.columns.values, loc= 'upper left')
plt.savefig('tmp.png')
plt.show()


# Show the daily simple returns
returns = df.pct_change()
returns.dropna(inplace=True)

#Show the annualised covariance matrix
cov_matrix_annual = returns.cov() * 252 #(relationship between two assets prices)

#Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))

#Calculate the portfolio volatility aka the sd
port_volatility = np.sqrt(port_variance)

#Claculate the annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights * 252)

#Show the expected annual return, volatility(risk), and variance

percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'
print('The expected annual return : '+ percent_ret)
print('The annual volatility / risk : '+ percent_vols)
print('The annual variance : '+ percent_var)

#Sharpe_ratio
sharpe_ratio = returns.mean() / returns.std()
sharpe_ratio.mean()



import pandas as pd
import numpy as np
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt import plotting

def deviation_risk_parity(w, cov_matrix):
    diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
    return (diff ** 2).sum().sum()


#Calculate the expected returns and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
ef = EfficientFrontier(mu, S)
weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
print(weights)
ef.portfolio_performance(verbose=True)

dfef = pd.DataFrame([weights])

#Plotting percentage of asstes allocation using EF
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

# Normalize result
result_pct = dfef.div(dfef.sum(1), axis=0)
ax = result_pct.plot(kind='bar',figsize=(15,4),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfef.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Efficient Frontier",fontsize= 16)
plt.xticks(fontsize=14)

for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')

# Hierarchical risk parity
hrp = HRPOpt(returns)
weights = hrp.optimize(linkage_method='single')
hrp.portfolio_performance(verbose=True)
print(weights)
plotting.plot_dendrogram(hrp)  # to plot dendrogram

dfhrp = pd.DataFrame([weights])

#Plotting percentage of asstes allocation using HRP
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']
# Normalize result
result_pct = dfhrp.div(dfhrp.sum(1), axis=0)
ax = result_pct.plot(kind='bar',figsize=(15,4),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfhrp.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Hierarchical Risk Parity",fontsize= 16)
plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')


# Crticial Line Algorithm
cla = CLA(mu, S)
weights = cla.max_sharpe()
print(weights)
cla.portfolio_performance(verbose=True)
plotting.plot_efficient_frontier(cla)  # to plot

dfcla = pd.DataFrame([weights])


#Plotting percentage of asstes allocation using CLA
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

# Normalize result
result_pct = dfcla.div(dfcla.sum(1), axis=0)

ax = result_pct.plot(kind='bar',figsize=(15,4),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfcla.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Critical Line Algorithm",fontsize= 16)

plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')


def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns) + 3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'sharpe'] + [asset for asset in assets])

    return results_df


mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)


#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation/volatility/risk
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
#create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Risk/Standard Deviation/Volatility')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
plt.show()


import scipy.optimize as sco
def calc_neg_sharpe(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return -sharpe_ratio
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
def max_sharpe_ratio(mean_returns, cov, rf):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, rf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_neg_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
optimal_port_sharpe = max_sharpe_ratio(mean_returns, cov, rf)

dfsr = pd.DataFrame([round(x,2) for x in optimal_port_sharpe['x']],index=assets).T

#Plotting percentage of asstes allocation using Scipy with maximum sharpe_ratio
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

# Normalize result
result_pct = dfsr.div(dfsr.sum(1), axis=0)

ax = result_pct.plot(kind='bar',figsize=(15,4),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfsr.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Scipy Max Sharpe_Ratio",fontsize= 16)

plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')

def calc_portfolio_std(weights, mean_returns, cov):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return portfolio_std
def min_variance(mean_returns, cov):
    num_assets = len(mean_returns)
    args = (mean_returns, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_std, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
min_port_variance = min_variance(mean_returns, cov)

dfmv = pd.DataFrame([round(x,2) for x in min_port_variance['x']],index=assets).T

#Plotting percentage of asstes allocation using Scipy with minimum variance
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

# Normalize result
result_pct = dfmv.div(dfmv.sum(1), axis=0)

ax = result_pct.plot(kind='bar',figsize=(15,6),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfmv.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Scipy Min Variance",fontsize= 16)

plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')

from scipy import stats


def calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, portfolio_var


def simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days):
    results_matrix = np.zeros((len(mean_returns) + 3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, portfolio_VaR = calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha,
                                                                                 days)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = portfolio_VaR
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'VaR'] + [asset for asset in assets])

    return results_df

mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
days = 252
alpha = 0.05
results_frame = simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days)

#locate positon of portfolio with minimum VaR(Value at Risque)
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
#create scatter plot coloured by VaR
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.VaR,results_frame.ret,c=results_frame.VaR,cmap='RdYlBu')
plt.xlabel('Value at Risk')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of minimum VaR portfolio
plt.scatter(min_VaR_port[2],min_VaR_port[0],marker=(5,1,0),color='r',s=500)
plt.show()

#locate positon of portfolio with minimum VaR
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
#create scatter plot coloured by VaR
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.VaR,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of minimum VaR portfolio
plt.scatter(min_VaR_port[1],min_VaR_port[0],marker=(5,1,0),color='r',s=500)
plt.show()

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
def calc_portfolio_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_var
def min_VaR(mean_returns, cov, alpha, days):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, alpha, days)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_VaR, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
min_port_VaR = min_VaR(mean_returns, cov, alpha, days)

dfvar = pd.DataFrame([round(x,2) for x in min_port_VaR['x']],index=assets).T

#Plotting percentage of asstes allocation using Scipy with minimum value at risk
colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

# Normalize result
result_pct = dfvar.div(dfvar.sum(1), axis=0)

ax = result_pct.plot(kind='bar',figsize=(15,6),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=dfvar.columns,fontsize= 14)
plt.title("Percentage of Assets Allocation using Scipy Min Value At Risk",fontsize= 16)

plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')



