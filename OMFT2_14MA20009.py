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
m = return_data.mean()
deviation_matrix = return_data.sub(return_data.mean())
row_count = return_data.count()[0]
cov_mat = 1 / float(row_count) * np.matmul(np.transpose(deviation_matrix), deviation_matrix)
w_mat = np.matmul(np.linalg.inv(cov_mat), np.ones(20)) / np.matmul(np.transpose(np.ones(20)), np.matmul(np.linalg.inv(cov_mat), np.ones(20)))

days_in_year = 365
u_f = 0.0575/days_in_year
print("uf = " + str(u_f) + "\n")

temp_mat = np.matmul(np.linalg.inv(cov_mat), (m - u_f * np.ones(20)))
w_d = temp_mat / np.matmul(np.transpose(np.ones(20)), temp_mat)

print("w_d values :")
print(w_d)
print("\nSum of w_d : " + str(sum(w_d)))
np.matmul(np.transpose(w_d), m) / np.mw_d, np.transpose(cov_mat)

