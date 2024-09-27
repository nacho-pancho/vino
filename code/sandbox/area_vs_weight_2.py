#!/usr/bin/env python3
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from matplotlib import cm

area_by_sector = pd.read_csv('data/vino_fino/pixel_area_by_sector.csv',delimiter=';',decimal=',')
weight_by_sector = pd.read_csv('data/vino_fino/lines_sectors_weights.csv')
sector_ids =  [ f'{a}-{b}' for a,b in zip (area_by_sector['Start'],area_by_sector['Finish'])]
print(sector_ids)
area_by_sector['sector_id'] = sector_ids
area_by_sector.set_index('sector_id',inplace=True)
weight_by_sector.set_index('sector_id',inplace=True)
area_by_sector.drop(['Start','Finish'],inplace=True,axis=1)
area_by_sector = area_by_sector.groupby('sector_id').sum()
area_by_sector.drop(['Cam'],inplace=True,axis=1)
area_and_weight_by_sector = area_by_sector.merge(weight_by_sector,how='inner',on='sector_id',sort=True)
weights = np.array(area_and_weight_by_sector['weight'])
maxX = 5
plt.figure(figsize=(9,9))
for j,expo in enumerate(np.linspace(0.1,1.0,10)):
    for i,measure in enumerate(['Min','Max','Mean','Median']):
        plt.subplot(2,2,i+1)
        area = np.array(area_and_weight_by_sector[measure])
        area = np.power(area/100000,expo).reshape(-1,1)
        robust = HuberRegressor(epsilon=2).fit(area,weights)
        linear = LinearRegression().fit(area,weights)
        area_inliers = area[~robust.outliers_]
        weights_inliers = weights[~robust.outliers_]
        rwhat = robust.predict(area_inliers)
        lwhat = linear.predict(area_inliers)
        rscore = np.sqrt(np.mean(np.square(rwhat-weights_inliers)))
        lscore = np.sqrt(np.mean(np.square(lwhat-weights_inliers)))
        print(f'measure {measure:6} exponent {expo:4.1f} robust {rscore:5.3f} linear {lscore:5.3f} outliers {robust.outliers_.sum()}')
        plt.scatter(area,weights,c=cm.seismic(robust.outliers_.astype(float)))
        x = np.linspace(0,100,2)
        yl = linear.predict(x.reshape(-1,1))
        yr = robust.predict(x.reshape(-1,1))
        plt.plot(x,yl,label=f'linear MSE={lscore:5.3f}')
        plt.plot(x,yr,label=f'robust MSE={rscore:5.3f}')
        plt.xlabel(f'{measure} area')
        plt.ylabel('weight')
        plt.title(f'{measure} vs area')
        plt.legend()
        plt.xlim(0,np.max(area))
        plt.ylim(0,np.max(weights))
        plt.grid(True)
    plt.savefig(f'area_vs_weight_{expo}.png')
    plt.close()
    