#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from matplotlib import cm
import csv
from numpy import random

with open('data/vino_fino/merged_sector_data_with_weights.csv','r') as f:
    reader = csv.reader(f)
    header = next(reader)
    #print(header)
    data = list(np.array(row[4:],dtype=float) for row in reader)
data = np.array(data)
data[:,:4] /= 100000
weights = data[:,-1]
n = len(weights)
plt.figure(figsize=(9,9))
rng = random.default_rng()
K = 30

for j,expo in enumerate(np.linspace(0.1,1.0,10)):
    for i,measure in enumerate(['Min','Max','Mean','Median']):
        plt.subplot(2,2,i+1)
        area = np.power(data[:,i],expo).reshape(-1,1)
        r_rmse = np.zeros(K)
        l_rmse = np.zeros(K)
        r_mape = np.zeros(K)
        l_mape = np.zeros(K)
        for k in range(K):
            idx = rng.uniform(0,1,n) < 0.9
            area_train = area[idx]
            weights_train = weights[idx]
            area_val = area[~idx]
            weights_val = weights[~idx]
            robust = HuberRegressor(epsilon=2,fit_intercept=False).fit(area_train,weights_train)
            linear = LinearRegression(fit_intercept=False).fit(area_train,weights_train)
            area_inliers = area_train[~robust.outliers_]
            weights_inliers = weights_train[~robust.outliers_]
            rwhat = robust.predict(area_inliers)
            lwhat = linear.predict(area_inliers)
            r_rmse[k] = np.sqrt(np.mean(np.square(rwhat-weights_inliers)))
            l_rmse[k] = np.sqrt(np.mean(np.square(lwhat-weights_inliers)))
            r_mape[k] = np.mean(np.abs(rwhat-weights_inliers)/weights_inliers)
            l_mape[k] = np.mean(np.abs(lwhat-weights_inliers)/weights_inliers)
        r_rmse_m, r_rmse_v = np.mean(r_rmse), np.std(r_rmse)
        l_rmse_m, l_rmse_v = np.mean(l_rmse), np.std(l_rmse)
        r_mape_m, r_mape_v = np.mean(r_mape), np.std(r_mape)
        l_mape_m, l_mape_v = np.mean(l_mape), np.std(l_mape)
        print(f'measure {measure:6} exponent {expo:4.1f}:')
        print(f'\trobust RMSE mean {r_rmse_m:5.3f} dev {r_rmse_v:5.3f}')
        print(f'\trobust MAPE mean {r_mape_m:5.3f} dev {r_mape_v:5.3f}')
        print(f'\tlinear RMSE mean {l_rmse_m:5.3f} dev {l_rmse_v:5.3f}')
        print(f'\tlinear MAPE mean {l_mape_m:5.3f} dev {l_mape_v:5.3f}')
        #plt.scatter(area,weights,c=cm.seismic(robust.outliers_.astype(float)))
        plt.scatter(area,weights)
        x = np.linspace(0,100,2)
        yl = linear.predict(x.reshape(-1,1))
        yr = robust.predict(x.reshape(-1,1))
        plt.plot(x,yl,label=f'linear MAPE={l_mape_m:5.3f}')
        plt.plot(x,yr,label=f'robust MAPE={r_mape_m:5.3f}')
        plt.xlabel(f'{measure} area')
        plt.ylabel('weight')
        plt.title(f'{measure} vs area')
        plt.legend()
        plt.xlim(0,np.max(area))
        plt.ylim(0,np.max(weights))
        plt.grid(True)
    plt.savefig(f'area_vs_weight_{expo}.png')
    plt.close()
    