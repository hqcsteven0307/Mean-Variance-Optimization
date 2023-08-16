import numpy as np
from scipy.optimize import minimize

import cvxopt
cvxopt.solvers.options['show_progress'] = False

def objective_expand(weights, ret, cov_matrix, alpha=5):
    num_assets = len(weights) // 3
    port_ret = np.dot(weights[:num_assets].T, ret).sum()
    port_risk = np.dot(weights[:num_assets].T, np.dot(cov_matrix, weights[:num_assets]))
    value = port_ret - 0.5 * alpha * port_risk
    return -value

def weight_constraint(weights, target_weight = 1):
    num_assets = len(weights) // 3
    return np.sum(weights[:num_assets]) - target_weight

def weight_eq(weights, prev_weights):
    num_assets = len(weights) // 3
    return weights[:num_assets] -(prev_weights + weights[num_assets:2*num_assets] - weights[2*num_assets:])

def turnover_constraint_expand(weights, max_change = 0.15):
    num_assets = len(weights) // 3
    return max_change - (weights[num_assets:2*num_assets].sum() + weights[2*num_assets:].sum())

def optimize_portfolio(expected_returns, cov_matrix, prev_weights, index_weights, target_weight=1, max_deviation = 0.03, max_turnover = 0.15):        
    num_assets = len(expected_returns)    
    initial_guess = np.array([0.0] * num_assets * 3)
    initial_guess[:num_assets] = prev_weights
            
    args = (expected_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: weight_constraint(x, target_weight)},
                   {'type': 'eq', 'fun': lambda x: weight_eq(x, prev_weights)},
                   {'type': 'ineq', 'fun': lambda x: turnover_constraint_expand(x, max_turnover)}
                   )

    bounds = list((max(0, index_weights[i] - 0.03), index_weights[i] + 0.03) for i in range(num_assets))
    
    positive_bound = list((0, 0.03) for i in range(num_assets))
    negative_bound = list((0, min(prev_weights[i], 0.03)) for i in range(num_assets))
    bounds_all = tuple(bounds + positive_bound + negative_bound)
    
    result = minimize(objective_expand, initial_guess, args=args, method='SLSQP', bounds=bounds_all, constraints=constraints, options={'maxiter': 1000})
    return result

def optimize_portfolio_cvxopt(expected_rets, cov_matrix, prev_weights, index_weights, target_weight=1, max_deviation = 0.03, max_turnover = 0.15):
    n = len(expected_rets)
    Q = cvxopt.matrix(np.block([
        [cov_matrix.values, np.zeros((n, n)), np.zeros((n, n))],
        [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))],
        [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))]
    ]))
    r = cvxopt.matrix(np.hstack((expected_rets.values, np.zeros(n), np.zeros(n))).reshape(-1, 1))
    
    A = cvxopt.matrix(np.block([
        [np.ones(n), np.zeros(n), np.zeros(n)],
        [np.eye(n), -np.eye(n), np.eye(n)]
        ]))
    b = cvxopt.matrix(np.block([[np.ones(1) * target_weight], [prev_weights.values.reshape(-1, 1)]]))
    
    G = cvxopt.matrix(np.block([
        [-np.eye(3 * n)],
        [np.zeros((1, n)), np.ones((1, n)), np.ones((1, n))],
        [np.eye(n), np.zeros((n, n)), np.zeros((n, n))],
        [-np.eye(n), np.zeros((n, n)), np.zeros((n, n))]
    ]))
    
    h = cvxopt.matrix(np.vstack([np.zeros((3*n, 1)), 
                                 np.array(max_turnover),
                                index_weights.values.reshape(-1, 1) + max_deviation,
                                max_deviation - index_weights.values.reshape(-1, 1),
                                 ]))
    
    sol = cvxopt.solvers.qp(5 * Q, -r, G, h, A, b)
    return sol
