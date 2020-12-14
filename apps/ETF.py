from pandas_datareader import data as web
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import HRPOpt


def app() :

    # Get the stock symbols in the portfolio
    # FAANG
    assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG', 'IBM', 'MSFT', 'MCD', 'MA', 'TSLA', 'KO', 'NVDA', 'HD', 'ORCL',
              'SAP',
              'CSCO', 'VMW', 'INTC', 'NOW', 'BLK', 'EBAY', 'DXC', 'VMC', 'VLO', 'WM', 'WST']

    st.title('Portfolio Diversification')

    st.write("""
    # Explore different optimizer and datasets
    """)

    ETF_name = st.sidebar.multiselect('Select ETF stocks :', assets)

    date_range = st.date_input(
    "Select date range",
    [datetime.date(2018, 1, 1), datetime.date.today()])

    x = st.slider('Select the date range', 2013, 2020, (2013, 2020))
    # 'Select the year range' -> Text to display
    # 1996 -> The lower bound
    # 2017 -> The higher bound
    # (1996, 2017) -> Default selected range




    # st.write(f"## {ETF_name} ETF")
    model_name = st.sidebar.selectbox(
        'Select optimizer',
        ('', 'EF', 'HRP', 'ScipyOpt')
    )
    # st.write(f"## {model_name} Model")

    weights = np.random.random(len(ETF_name))
    weights = weights / np.sum(weights)

    if len(date_range) == 2 :
        # Get the stocks/portfolio starting date
        stockStartDate = str(date_range[0].strftime('%Y-%m-%d'))

        # Get the stocks ending date (today)
        today = str(date_range[1].strftime('%Y-%m-%d'))




    # Create a dataframe to store adjusted close price of the stocks
    df_ETF = pd.DataFrame()

    # Store the adjusted close price into the df
    for stock in ETF_name:
        df_ETF[stock] = web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']

    # Show the daily simple returns
    returns = df_ETF.pct_change()
    returns.dropna(inplace=True)

    ############################################# Efficient Frontier #######################################################

    def deviation_risk_parity(w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    def calculEF(dataframe):
        mu = expected_returns.mean_historical_return(dataframe)
        S = risk_models.sample_cov(dataframe)
        ef = EfficientFrontier(mu, S)
        weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
        ef.portfolio_performance(verbose=True)
        return pd.DataFrame([weights])

    colors_list = ['#5cb85c', '#F9429E', '#2C75FF', '#DF73FF', '#25FDE9', '#660099']

    if len(ETF_name) != 0:

        plt.style.use('seaborn-white')
        df_ETF.plot(legend=0, figsize=(10, 6), grid=True, title='Daily Returns of the ETF Stocks')
        plt.tight_layout()
        plt.ylabel('Adj. Price USD ($)', fontsize=10)
        plt.legend(df_ETF.columns.values, loc='upper left')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.balloons()


    if model_name == 'EF':
        if len(ETF_name) == 0:
            st.write("Please select stocks!!!")
        if len(ETF_name) != 0:
            plt.clf()
            df_ETF = calculEF(df_ETF)
            result_pct = df_ETF.div(df_ETF.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
            plt.legend(labels=df_ETF.columns, fontsize=20)
            plt.xticks(fontsize=20)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.yticks([])
            # Add this loop to add the annotations
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate(f'{height:.0%}', (x + width / 2, y + height), ha='center', fontsize=20)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(f"## ETF Assets Allocation using Efficient Frontier")
            st.pyplot()

    ################################## Herarchical Risk Parity ###############################################################

    if model_name == 'HRP':

        hrp_ETF = HRPOpt(returns)

        weights = hrp_ETF.optimize(linkage_method='single')

        hrp_ETF.portfolio_performance(verbose=True)

        dfhrpETF = pd.DataFrame([weights])

        if len(ETF_name) == 0:
            st.write("Please select stocks!!!")
        if len(ETF_name) != 0:
            result_pct = dfhrpETF.div(dfhrpETF.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
            plt.legend(labels=dfhrpETF.columns, fontsize=20)
            plt.xticks(fontsize=20)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.yticks([])
            # Add this loop to add the annotations
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center', fontsize=20)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(f"## ETF Assets Allocation using Hierarchical Risk Parity")
            st.pyplot()

    ############################################ Scipy Optimize #############################################################



    if model_name == 'ScipyOpt':

        if len(ETF_name) == 0:
            st.write("Please select stocks!!!")

        mean_returns_ETF = df_ETF.pct_change().mean()
        cov_ETF = df_ETF.pct_change().cov()
        rf = 0.0

        import scipy.optimize as sco
        def calc_neg_sharpe(weights, mean_returns, cov, rf):
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - rf) / portfolio_std
            return -sharpe_ratio

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def max_sharpe_ratio(mean_returns, cov, rf):
            result = 0
            num_assets = len(mean_returns)
            args = (mean_returns, cov, rf)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            if num_assets != 0:
                result = sco.minimize(calc_neg_sharpe, num_assets * [1. / num_assets, ], args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)

            return result

        optimal_port_sharpe_ETF = max_sharpe_ratio(mean_returns_ETF, cov_ETF, rf)
        dfsETF = pd.DataFrame([round(x, 2) for x in optimal_port_sharpe_ETF['x']], index=ETF_name).T


        if len(ETF_name) != 0:
            result_pct = dfsETF.div(dfsETF.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
            plt.legend(labels=dfsETF.columns, fontsize=20)
            plt.xticks(fontsize=20)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.yticks([])
            # Add this loop to add the annotations
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center', fontsize=20)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(f"## ETF Assets Allocation using Scipy Opt")
            st.pyplot()

