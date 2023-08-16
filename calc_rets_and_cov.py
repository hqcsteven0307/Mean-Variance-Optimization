import datetime
from config import exclude_codes
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from statsmodels.stats.correlation_tools import cov_nearest

def get_factor_rets(date, 
                    history, 
                    weights, 
                    norm_factor_exposure, 
                    industry_stock):
    # get the stock data
    nonzero_codes = weights.loc[date][weights.loc[date] != 0].index.tolist()
    nonzero_codes = sorted(set(nonzero_codes) - set(exclude_codes))
    
    full_history_ret = history[history.index == date].loc[:, ['code', 'ret']].set_index(['code'])
    
    history_ret = full_history_ret.loc[nonzero_codes,:]
    history_ret = history_ret.loc[~history_ret['ret'].isna(),:]
    nonzero_codes = history_ret.index.tolist()
        
    full_risk_factors = norm_factor_exposure[norm_factor_exposure.index == date].set_index('code')
    risk_factors = full_risk_factors.loc[nonzero_codes,:]
    risk_factors = risk_factors.dropna(how='all', axis=1)
    
    industry_factors = pd.get_dummies(industry_stock.loc[datetime.datetime(date.year, 1, 1),nonzero_codes])
    country_factor = pd.DataFrame(np.ones(history_ret.shape[0]), index=history_ret.index, columns=['country'])
    factors_all = pd.concat([country_factor, risk_factors, industry_factors], axis=1)
    
    full_factors_all = pd.DataFrame(index=weights.columns, columns=factors_all.columns)
    full_factors_all['country'] = 1
    full_factors_all.loc[:, risk_factors.columns] = full_risk_factors
    full_factors_all.loc[:, industry_factors.columns] = pd.get_dummies(industry_stock.loc[datetime.datetime(date.year, 1, 1),:])[industry_factors.columns]
    
    # get the hisory market cap for the WLS and constraints
    history_cap = history[history.index == date].loc[:, ['code', 'market_cap']].set_index('code')
    history_cap = history_cap.loc[nonzero_codes, :]
    history_cap_sqrt_weight = (np.sqrt(history_cap) / np.sqrt(history_cap).sum())
    history_cap_weight = history_cap / history_cap.sum()
    weights_reg = np.array(history_cap_sqrt_weight['market_cap'].tolist())

    industry_date = industry_stock.loc[datetime.datetime(date.year, 1, 1),nonzero_codes].rename('industry').to_frame()

    weights_industry = pd.merge(industry_date, history_cap_weight, how='left', left_index=True, right_index=True).groupby('industry')['market_cap'].sum().tolist()
    
    # add the constraints, the weighted sum of the industry returns should be 0 to avoid the multicollinearity
    A_eq = np.zeros((1, factors_all.shape[1]))
    A_eq[0, (1+risk_factors.shape[1]):] = weights_industry
    b_eq = np.array([0.0])
    constraint = LinearConstraint(A_eq, b_eq, b_eq)
    
    # define the objective and constraints for the optimization
    X = factors_all.values
    y = history_ret['ret'].values
    
    def objective(params):
        return np.sum(weights_reg * (y - X.dot(params))**2)

    initial_guess = np.zeros(X.shape[1])
    result = minimize(objective, initial_guess, constraints=constraint)
    
    factor_ret = pd.DataFrame(result.x, index = factors_all.columns, columns = [date]).T
    
    residual_ret = (full_factors_all.dot(factor_ret.T) - full_history_ret.rename(columns={'ret':date})).T
    
    # get the r square of the regression
    y_pred = X.dot(result.x)
    weighted_mean = y.dot(weights_reg)
    
    TSS = np.sum(weights_reg * (y - weighted_mean)**2)
    RSS = np.sum(weights_reg * (y - y_pred)**2)
    ESS = np.sum(weights_reg * (y_pred - weighted_mean)**2)

    r_square = ESS / TSS
        
    return factor_ret, residual_ret, r_square

def get_cov_date(date, 
                 factor_cov, 
                 factor_exposure, 
                 residual_vol, 
                 weights,
                 industry_stock,
                 codes = None):
    factor_cov_date = factor_cov[factor_cov.index.get_level_values(0) == date].droplevel(0)
    factor_cov_date = factor_cov_date.dropna(axis=1, how='all').dropna(axis=0, how='all')
    factor_cov_date = pd.DataFrame(cov_nearest(factor_cov_date), index=factor_cov_date.index, columns=factor_cov_date.columns)
    residual_vol_date = residual_vol[residual_vol.index == date]

    factor_exposure_date = factor_exposure[factor_exposure.index.get_level_values(0) == date].set_index('code').dropna(how='all', axis=1)
    industry_factor_date = pd.get_dummies(industry_stock.loc[datetime.datetime(date.year, 1, 1), :])
    industry_factor_date.columns = industry_factor_date.columns.astype(str)
    industry_factor_date = industry_factor_date.loc[:,industry_factor_date.columns.isin(factor_cov_date.columns)]

    country_factor_date = pd.DataFrame(1, index=factor_exposure_date.index, columns=['country'])

    if codes is None:
        nonzero_codes = weights.loc[date][weights.loc[date] != 0].index.tolist()
    else:
        nonzero_codes = codes

    beta_date = pd.concat([
        country_factor_date.loc[nonzero_codes,:],
        factor_exposure_date.loc[nonzero_codes,:], 
        industry_factor_date.loc[nonzero_codes,:]], axis=1)    
    
    factor_cov_date = factor_cov_date.fillna(0)
    
    if beta_date.shape[1] != factor_cov_date.shape[0]:
        # in some case we do not have the components but the cov is still existing, then we ignore those factor cov
        asset_cov_date = beta_date.dot(factor_cov_date.loc[beta_date.columns, beta_date.columns]).dot(beta_date.T) + \
            np.diag(residual_vol_date.loc[date, nonzero_codes].fillna(residual_vol_date.loc[date, nonzero_codes].mean()) ** 2)            
    else:        
        asset_cov_date = beta_date.dot(factor_cov_date).dot(beta_date.T) + np.diag(residual_vol_date.loc[date, nonzero_codes].fillna(residual_vol_date.loc[date, nonzero_codes].mean()) ** 2)    
    return asset_cov_date