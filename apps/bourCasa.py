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



def app():


    df_morocco = pd.read_excel('Bourse_De_Casa.xlsx')
    df_morocco.set_index('Session', inplace=True)

    st.title('Portfolio Diversification')

    st.write("""
    # Explore different optimizer and datasets
    """)

    # moroccan_stocks = st.sidebar.multiselect('Select moroccan stocks :', df_morocco.columns.difference(['Session']))
    moroccan_stocks = st.sidebar.multiselect('Select moroccan stocks :', df_morocco.columns)
    df_moroccan_stocks = df_morocco.loc[:, df_morocco.columns.isin(moroccan_stocks)]

    model_name = st.sidebar.selectbox(
        'Select optimizer',
        ('', 'EF', 'HRP', 'ScipyOpt')
    )

    date_range = st.date_input(
    "Select date range",
    [datetime.date.today(), datetime.date.today()])

    wei = np.random.random(len(moroccan_stocks))
    wei = wei / np.sum(wei)

    # Get the stocks/portfolio starting date
    if len(date_range) == 2 :
        stockStartDate = str(date_range[0].strftime('%Y-%m-%d'))
        # Get the stocks ending date (today)
        today = str(date_range[1].strftime('%Y-%m-%d'))
        # df_moroccan_stocks = (df_moroccan_stocks['Session'] <= today) & (df_moroccan_stocks['Session'] >= stockStartDate)
        df_moroccan_stocks = df_moroccan_stocks.loc[stockStartDate:today]
        st.write(df_moroccan_stocks)



    moroccan_stocks_returns = df_moroccan_stocks.pct_change()
    moroccan_stocks_returns.dropna(inplace=True)

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

    if len(moroccan_stocks) != 0:
        plt.style.use('seaborn-white')
        df_moroccan_stocks.plot(legend=0, figsize=(10, 6), grid=True, title='Daily Returns of moroccan Stocks')
        plt.tight_layout()
        plt.ylabel('Adj. Price USD ($)', fontsize=10)
        plt.legend(df_moroccan_stocks.columns.values, loc='upper left')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.balloons()


    if model_name == 'EF':

        if len(moroccan_stocks) == 0:
            st.write("Please select stocks!!!")

        if len(moroccan_stocks) != 0:
            df_moroccan_stocks = calculEF(df_moroccan_stocks)
            result_pct = df_moroccan_stocks.div(df_moroccan_stocks.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
            plt.legend(labels=df_moroccan_stocks.columns, fontsize=20)
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
            st.write(f"## Moroccan Assets Allocation using Efficient Frontier")
            st.pyplot()

    ################################## Herarchical Risk Parity ###############################################################

    if model_name == 'HRP':


        hrp_Mor = HRPOpt(moroccan_stocks_returns)

        wei = hrp_Mor.optimize(linkage_method='single')

        hrp_Mor.portfolio_performance(verbose=True)

        dfhrpMor = pd.DataFrame([wei])

        if len(moroccan_stocks) == 0:
            st.write("Please select stocks!!!")
        if len(moroccan_stocks) != 0:
            result_pct = dfhrpMor.div(dfhrpMor.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
            plt.legend(labels=dfhrpMor.columns, fontsize=20)
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
            st.write(f"## Moroccan Assets Allocation using Hierarchical Risk Parity")
            st.pyplot()

    ############################################ Scipy Optimize #############################################################



    if model_name == 'ScipyOpt':


        rf = 0.0
        mean_returns_M = df_moroccan_stocks.pct_change().mean()
        cov_M = df_moroccan_stocks.pct_change().cov()

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
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(calc_neg_sharpe, num_assets * [1. / num_assets, ], args=args,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
            return result

        optimal_port_sharpe_M = max_sharpe_ratio(mean_returns_M, cov_M, rf)

        dfsM = pd.DataFrame([round(x, 2) for x in optimal_port_sharpe_M['x']], index=moroccan_stocks).T

        if len(moroccan_stocks) == 0:
            st.write("Please select stocks!!!")

        if len(moroccan_stocks) != 0:
            result_pct = dfsM.div(dfsM.sum(1), axis=0)
            ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
            plt.legend(labels=dfsM.columns, fontsize=20)
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
            st.write(f"## Moroccan Assets Allocation using ScipyOpt")
            st.pyplot()



