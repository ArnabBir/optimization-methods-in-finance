import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getMarkowitz2Weights(return_data, R):
    return_mean = return_data.mean()
    cov_inv_mean = np.matmul(np.linalg.inv(cov_mat), return_mean)
    cov_inv_unit = np.matmul(np.linalg.inv(cov_mat), np.ones(20))
    m11 = np.matmul(np.transpose(return_mean), cov_inv_mean)
    m12 = np.matmul(np.transpose(return_mean), cov_inv_unit)
    m21 = np.matmul(np.transpose(np.ones(20)), cov_inv_mean)
    m22 = np.matmul(np.transpose(np.ones(20)), cov_inv_unit)
    lambda1 = np.linalg.det(np.matrix([[2*R, m12], [2, m22]])) / np.linalg.det(np.matrix([[m11, m12], [m21, m22]]))
    lambda2 = np.linalg.det(np.matrix([[m11, 2*R], [m21, 2]])) / np.linalg.det(np.matrix([[m11, m12], [m21, m22]]))
    weights = lambda1 * cov_inv_mean + lambda2 * cov_inv_unit
    return weights

def getMarkowitz2VarianceRisk(return_data, R):
    weights = getMarkowitz2Weights(return_data, R)
    variance_risk = np.matmul( np.transpose(weights), np.matmul(np.transpose(cov_mat), weights)) 
    return variance_risk

return_data = pd.read_csv("asset_return_data.csv").drop('Id', 1)
deviation_matrix = return_data.sub(return_data.mean())
cov_mat = 1 / 250.0 * np.matmul(np.transpose(deviation_matrix), deviation_matrix)
R = 0.02

weights = getMarkowitz2Weights(return_data, 0.02)

R = 0.01
R_list = []
variance_risk_list = []
while(R < 0.9):
    variance_risk_list.append(getMarkowitz2VarianceRisk(return_data, R))
    R_list.append(R)
    R += 0.01
#len(variance_risk_list) / len(R_list)
plt.plot(variance_risk_list, R_list)
plt.show()