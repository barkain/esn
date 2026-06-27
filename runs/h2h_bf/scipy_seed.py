import numpy as np
import scipy.optimize as opt


def objective(params):
    radii = params[52:]
    return -np.sum(radii)


def constraints(params):
    centers = params[:52].reshape((26, 2))
    radii = params[52:]
    cons = []
    for i in range(25):
        for j in range(i + 1, 26):
            cons.append((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2 - (radii[i] + radii[j])**2)
    for k in range(26):
        cons.append(centers[k][0] - radii[k])
        cons.append(centers[k][1] - radii[k])
        cons.append(1 - centers[k][0] - radii[k])
        cons.append(1 - centers[k][1] - radii[k])
    return cons


def solve():
    n = 26
    initial_centers = np.random.rand(n, 2) * 0.8 + 0.1
    initial_radii = np.random.rand(n) * 0.05 + 0.01
    initial_params = np.concatenate([initial_centers.flatten(), initial_radii])
    bounds = [(0, 1) for _ in range(52)] + [(1e-5, 0.25) for _ in range(n)]
    constraints_obj = {'type': 'ineq', 'fun': constraints}
    result = opt.minimize(objective, initial_params, bounds=bounds, constraints=constraints_obj)
    p = result.x
    return p[:52].reshape((n, 2)), p[52:]
