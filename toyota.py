# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:31:34 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

corolla_data=pd.read_csv('C:/Users/HP/Desktop/assignments submission/multi linear regression/ToyotaCorolla.csv',encoding="ISO-8859-1")
corolla_data.columns
Toyota=corolla_data[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
Toyota.columns
Toyota.isna().sum()
Toyota.dropna()

#################EDA#############
import seaborn as sns
sns.pairplot(Toyota)

Toyota.corr()

#####################Model building##################
import statsmodels.formula.api as smf
toyota_model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit()
toyota_model.params
toyota_model.summary()
#R_sq=0.864
#cc,Doors are not significant

###############Validation Techniques##########
#VIF
age_rsq=smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_age=1/(1-age_rsq)

KM_rsq=smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_KM=1/(1-KM_rsq)

HP_rsq=smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_HP=1/(1-HP_rsq)

cc_rsq=smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_cc=1/(1-cc_rsq)

doors_rsq=smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_doors=1/(1-doors_rsq)

gears_rsq=smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=Toyota).fit().rsquared
VIF_gears=1/(1-gears_rsq)

tax_rsq=smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=Toyota).fit().rsquared
VIF_tax=1/(1-tax_rsq)

weight_rsq=smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Gears',data=Toyota).fit().rsquared
VIF_weight=1/(1-weight_rsq)

d1={'variables':['age','KM','HP','cc','doors','gears','tax','weight'],'VIF':[VIF_age,VIF_KM,VIF_HP,VIF_cc,VIF_doors,VIF_gears,VIF_tax,VIF_weight]}
VIF_frame=pd.DataFrame(d1)
VIF_frame
#VIF>20 then collinearity


#################Validation plots###################

import statsmodels.api as sm
sm.graphics.plot_partregress_grid(toyota_model)

#############Deletion Diagnostic###############

sm.graphics.influence_plot(toyota_model)
#80

#################Iteration1##################
Toyota1=Toyota.drop(Toyota.index[80],axis=0)
toyota_model1=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota1).fit()
toyota_model1.params
toyota_model1.summary()
#R_sq=0.869
#doors is not significant
sm.graphics.plot_partregress_grid(toyota_model1)
sm.graphics.influence_plot(toyota_model1)
#221

#################Iteration2,3,4##################
Toyota2=Toyota.drop(Toyota.index[[80,221,960,601]],axis=0)
toyota_model2=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota2).fit()
toyota_model2.params
toyota_model2.summary()
#R_sq=0.878,0.885,0.889
#1. doors nor significant,2.all are significant
sm.graphics.plot_partregress_grid(toyota_model2)
sm.graphics.influence_plot(toyota_model2)
#960,601

#####################Transformations##################
Toyota2=Toyota.drop(Toyota.index[[80,221,960,601]],axis=0)
toyota_model3=smf.ols('Price~Age_08_04+KM+HP+np.log(cc)+np.log(Doors)+Gears+Quarterly_Tax+Weight',data=Toyota2).fit()
toyota_model3.params
toyota_model3.summary()
#R_sq=0.89


Toyota2=Toyota.drop(Toyota.index[[80,221,960,601]],axis=0)
Toyota['cc_sq']=Toyota.cc*Toyota.cc
Toyota['Door_sq']=Toyota.Doors*Toyota.Doors
toyota_model4=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight+Door_sq',data=Toyota2).fit()
toyota_model4.params
toyota_model4.summary()
#R_sq=0.891
#all are significant

Toyota_predict=toyota_model4.predict(Toyota2)
Toyota_predict
Toyota_error=Toyota2.Price-Toyota_predict
Toyota_error
from sklearn.metrics import mean_squared_error
from math import sqrt
Toyota_rmse=sqrt(mean_squared_error(Toyota2.Price,Toyota_predict))
Toyota_rmse
#1195.49