import pandas as pd
import os
import numpy as np
from config import *
from calc_rets_and_cov import get_cov_date

def get_history_data(data_path = './data', volume_to_na=True):
    df = pd.read_csv(os.path.join(data_path, 'history.csv'), index_col=0, parse_dates=True)
    if volume_to_na:
        df.loc[df.volume == 0, 'close'] = np.nan
    
    return df

def get_market_cap(data_path = './data'):
    df = pd.read_csv(os.path.join(data_path, 'market_cap.csv'), index_col=0, parse_dates=True)
    return df

def get_weights(data_path = './data'):
    df = pd.read_csv(os.path.join(data_path, 'components.csv'), index_col=0, parse_dates=True)
    return df

def read_data(file_name, data_path = './data'):
    df = pd.read_csv(os.path.join(data_path, file_name), index_col=0, parse_dates=True)
    return df

def fill_weights(data_path = './data', history=None):
    df = get_weights(data_path)
    if history is None:
        history = get_history_data(data_path)
        history = history[history.index >= df.index.min()]
    components_all = pd.DataFrame(index=history.index.unique())
    components_all = components_all.merge(df, how='left', left_index=True, right_index=True)
    components_all.fillna(method='ffill', inplace=True)

    return components_all

def get_normalized_factor(factor_df, weights):
    """
    Normalize factor_df by weights
    """
    factor_df = factor_df.copy()
    ind_weights = (weights != 0).astype(int)
    ind_df = (ind_weights * factor_df)
    factor_mean = ind_df.mean(axis=1, skipna=True)
    factor_std = ind_df.std(axis=1, skipna=True)
    norm_df = factor_df.sub(factor_mean, axis=0).div(factor_std, axis=0)
    norm_df.clip(-3, 3, inplace=True)
    return norm_df

def get_shift_trading_date(date, trading_dates, shift):
    if shift == 0:
        return date    
    date_index = trading_dates.index(date)
    if date_index + shift >= len(trading_dates):
        return trading_dates[-1]
    elif date_index + shift < 0:
        return trading_dates[0]
    else:
        return trading_dates[date_index + shift]
    
def filter_target_stock(date, history, weights):
    nonzero_codes = weights.loc[date][weights.loc[date] != 0].index.tolist()
    for exclude_code in exclude_codes:
        if exclude_code in nonzero_codes:
            nonzero_codes.remove(exclude_code)

    next_date = get_shift_trading_date(date, weights.index.tolist(), 1)        
    history_ret = history[history.index == next_date].loc[:, ['code', 'ret']].set_index(['code'])
    
    # TODO
    # we just do not trade those stocks with return > 10% or < -10% on the next date as untradable
    # better handle is can buy for ret<-10% and sell for ret>10%    
    limit_sell_tickers = history_ret[round(history_ret.ret, 3) <= -0.1].index
    limit_buy_tickers = history_ret[round(history_ret.ret, 3) >= 0.1].index
    na_tickers = history_ret.loc[history_ret.ret.isna(),:].index

    tradable_codes = sorted(list(set(nonzero_codes) - 
                                 (set(na_tickers) | set(limit_sell_tickers) | set(limit_buy_tickers))))
    return tradable_codes

def prepare_data_for_optimization(date, 
                                  history,
                                  weights,
                                  weights_optimize_df,
                                  factor_cov,
                                  norm_factor_exposure,
                                  residual_vol,
                                  industry_stock,
                                  expected_rets_all):
    tradable_codes = filter_target_stock(date, history, weights)
    nonzero_codes = weights.loc[date][weights.loc[date] != 0].index.tolist()
    
    date_index = weights.index.get_loc(date)
    prev_nonzero_codes = weights.iloc[date_index-1]
    prev_nonzero_codes = prev_nonzero_codes[prev_nonzero_codes != 0].index.tolist()

    added_stock, remove_stock = set(), set()
    if nonzero_codes == prev_nonzero_codes:
        pass
    else:
        added_stock = set(nonzero_codes) - set(prev_nonzero_codes) - set(exclude_codes)
        remove_stock = set(prev_nonzero_codes) - set(nonzero_codes) - set(exclude_codes)        
                
    nonzero_codes = sorted(list(set(nonzero_codes) - set(exclude_codes)))        
    non_tradable_codes = sorted(list(set(nonzero_codes) - set(tradable_codes)))            
    
    initial_weight = pd.Series(np.ones(len(nonzero_codes))/len(nonzero_codes), index=nonzero_codes)
    index_weight = weights.loc[date, nonzero_codes]
    
    date_index = weights_optimize_df.index.get_loc(date)
    if date_index == 0:
        previous_weight = initial_weight
    else:
        previous_weight = weights_optimize_df.iloc[date_index-1][nonzero_codes]

    factor_cov_date = get_cov_date(date, 
                                   factor_cov, 
                                   norm_factor_exposure, 
                                   residual_vol, 
                                   weights,
                                   industry_stock,
                                   nonzero_codes)
    
    expected_rets = expected_rets.fillna(0)

    # only select the cov and target ret for tradable codes
    factor_cov_date = factor_cov_date.loc[tradable_codes, tradable_codes]
    expected_rets = expected_rets.loc[tradable_codes]

    weight_target = previous_weight.loc[tradable_codes].sum()
    return (factor_cov_date, expected_rets, weight_target, nonzero_codes, tradable_codes, added_stock, remove_stock, non_tradable_codes, date_index, previous_weight, index_weight)

def get_ewma_weights(length, halflife):
    # Calculate decay factor
    decay = np.exp(np.log(0.5) / halflife)

    # Calculate EWMA weights
    weights = np.power(decay, np.arange(length - 1, -1, -1))

    # Normalize weights to sum to 1
    weights /= weights.sum()
    return weights




    
    