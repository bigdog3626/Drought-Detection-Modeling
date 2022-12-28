import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def collect_error_metrics(data, true_col, pred_col):

    RMSE = mean_squared_error(y_true=data[true_col],y_pred=data[pred_col])
    MAE = mean_absolute_error(y_true=data[true_col], y_pred=data[pred_col])
    MAPE = mean_absolute_percentage_error(y_true=data[true_col], y_pred=data[pred_col])


    return RMSE, MAE, MAPE

def mm_to_inches(dFrame, col):
    dFrame[col] = dFrame[col] / 25.4
    return dFrame


def fit_gamma(data):
    shape, loc, scale = stats.gamma.fit(data)
    return shape, loc, scale


def spi_percip_graphs(dFrame, percip_col, x, pdf):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Distribution of Percipitation values
    sns.histplot(dFrame[percip_col].unique(), kde=True, ax=ax[0, 0])
    ax[0, 0].set_xlabel("PRCP")
    ax[0, 0].set_ylabel("Concentration")
    ax[0, 0].set_title("Distribution of Precipitation Values")

    """
    Now plot the probability density function via fitting to the gamma function
    Alpha: 2
    Beta: 1
    """

    ax[0, 1].plot(x, pdf)
    ax[0, 1].set_xlabel('PRCP')
    ax[0, 1].set_ylabel('GAMMA Distribution')
    ax[0, 1].set_title('PRCP distribution Fitted to Gamma')

    """
    Plot the estimated cumulative density function
    """
    sns.ecdfplot(dFrame[percip_col].unique(), legend=True, ax=ax[1, 0])
    ax[1, 0].set_title("Estimated Cummulative Density Function of PRCP")
    ax[1, 0].set_ylabel("Cummulative Proportion")
    ax[1, 0].set_xlabel("PRCP")

    new_x = np.linspace(-3, 3, len(dFrame))
    cdf = stats.norm.cdf(
        new_x, pdf
    )

    ax[1, 1].plot(new_x, cdf)
    ax[1, 1].set_xlabel('SPI')
    ax[1, 1].set_ylabel('Cummulative Probability')
    ax[1, 1].set_title('Standard Percipitation Index Threshhold')

    return fig, ax


def find_nearest(spi_dict, value):
    """
    Finds the nearest value in an array to a passed in value
    :param spi_dict: dict of ecf : SPI
    :param value: value to find nearest to
    :return: nearest res
    """
    res = spi_dict.get(value) or spi_dict[min(spi_dict.keys(), key=lambda key: abs(key - value))]
    return res


def find_nearest_spi(spi_cdf, v):
    res = find_nearest(spi_cdf, v)
    np.around(res, 3)
    return res


def get_spi(df, percip_col, pdf):
    data = df[percip_col].unique()
    ecdf = ECDF(data)
    new_x = np.linspace(-3, 3, len(df))
    cdf = stats.norm.cdf(
        new_x, pdf
    )
    spi_cdf = dict(zip(np.around(cdf, 8), new_x))
    spi_cdf[1.0] = 3
    new_col = f'{percip_col}_ECDF'

    df[new_col] = np.around(ecdf(df[percip_col]), 8)

    df['SPI'] = df.apply(lambda row: find_nearest_spi(spi_cdf, row[new_col]), axis=1)
    return df


def get_spi_from_precip_col(dFrame, percip_col):
    dFrame = mm_to_inches(dFrame, percip_col)
    data = dFrame[percip_col].sort_values().values
    shape, loc, scale = fit_gamma(data)
    x = np.linspace(0, 10, len(dFrame))
    pdf = stats.gamma.pdf(x, 3, loc, .5)
    fig, ax = spi_percip_graphs(dFrame, percip_col, x, pdf)
    dFrame = get_spi(dFrame, percip_col, pdf)

    return dFrame, fig, ax


def get_model_acc_via_r2(y_test, y_pred_test):
    score = r2_score(y_test, y_pred_test)
    return round(score, 2) * 100
