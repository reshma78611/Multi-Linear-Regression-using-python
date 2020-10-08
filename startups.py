# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:22:50 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startups=pd.read_csv('C:/Users/HP/Desktop/assignments submission/multi linear regression/50_Startups.csv')
state=pd.get_dummies(startups.State)
start_up=startups.drop('State',axis=1)
start_up.columns
start_up=pd.concat([start_up,state],axis=1)

start_up.columns=['RD','admin','marketing','profit','california','florida','newyork']
start_up
start_up.isna().sum()
start_up.dropna()

###################EDA###########################
import seaborn as sns
sns.pairplot(start_up)
start_up.corr()

###################Model Building#####################
import statsmodels.formula.api as smf
model=smf.ols('profit~RD+admin+marketing+california+florida+newyork',data=start_up).fit()
model.params
model.summary()
#R_sq=0.951
#admin,marketing are insignificant

###################Validation techniques################
#check for collinearity using VIF(Variance inflation factor)
rsq_RD=smf.ols('RD~admin+marketing+california+florida+newyork',data=start_up).fit().rsquared
VIF_RD=1/(1-rsq_RD)

rsq_admin=smf.ols('admin~RD+marketing+california+florida+newyork',data=start_up).fit().rsquared
VIF_admin=1/(1-rsq_admin)

rsq_mark=smf.ols('marketing~admin+RD+california+florida+newyork',data=start_up).fit().rsquared
VIF_mark=1/(1-rsq_mark)

rsq_cal=smf.ols('california~admin+RD+marketing+florida+newyork',data=start_up).fit().rsquared
VIF_cal=1/(1-rsq_cal)

rsq_florida=smf.ols('florida~admin+RD+california+marketing+newyork',data=start_up).fit().rsquared
VIF_florida=1/(1-rsq_florida)

rsq_newyork=smf.ols('newyork~admin+RD+california+florida+marketing',data=start_up).fit().rsquared
VIF_newyork=1/(1-rsq_newyork)

d1={'varaibles':['RD','admin','marketing','california','florida','newyork'],'VIF':[VIF_RD,VIF_admin,VIF_mark,VIF_cal,VIF_florida,VIF_newyork]}
VIF_frame=pd.DataFrame(d1)
VIF_frame
#if vif>20 then collinearity


####################Validation plots##################

import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model)

#######################Deletion Diagnostic#################
sm.graphics.influence_plot(model)
#49

#############################  Iteration 1  #######################

start_up1=start_up.drop(start_up.index[49],axis=0)
#49th  observation is deleted, now build new model with this data
model1=smf.ols('profit~RD+admin+marketing+california+florida+newyork',data=start_up1).fit()
model1.params
model1.summary()
#0.962
#admin,marketing are insignificant
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model1)
sm.graphics.influence_plot(model1)


##############################  Iteration 2  #######################
start_up2=start_up.drop(start_up.index[[49,48]],axis=0)
#48th  observation is deleted, now build new model with this data
model2=smf.ols('profit~RD+admin+marketing+california+florida+newyork',data=start_up2).fit()
model2.params
model2.summary()
#0.963
#admin,marketing are insignificant
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)


#############################  Iteration 3 with transformation  #######################
#apply transformation with sq of admin and marketing
start_up['admin_sq']=start_up.admin*start_up.admin
start_up['market_sq']=start_up.marketing*start_up.marketing
start_up3=start_up.drop(start_up.index[[49,48,46]],axis=0)
#46th  observation is deleted, now build new model with this data
model3=smf.ols('profit~RD+admin+marketing+california+florida+newyork+admin_sq+market_sq',data=start_up3).fit()
model3.params
model3.summary()
#0.963
#admin,marketing,admin_sq,market_sq are not significant
#with adding transformation even not significant
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model3)
sm.graphics.influence_plot(model3)

#############################  Iteration 4 with removing marketing  #######################
start_up3=start_up.drop(start_up.index[[49,48,46]],axis=0)
#after all possibilities we remove marketing
model4=smf.ols('profit~RD+admin+california+florida+newyork+admin_sq+market_sq',data=start_up3).fit()
model4.params
model4.summary()
#0.963
##admin,admin_sq,market_sq are not significant
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model4)
sm.graphics.influence_plot(model4)

#############################  Iteration 5 with removing admin  #######################
start_up3=start_up.drop(start_up.index[[49,48,46]],axis=0)
#still not significant so try with removing admin
model5=smf.ols('profit~RD+marketing+california+florida+newyork',data=start_up3).fit()
model5.params
model5.summary()
#0.96
#all are significant
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model5)
sm.graphics.influence_plot(model5)
# this is the best model

start_pred=model.predict(start_up)
start_pred
Error=start_up.profit-start_pred
Error

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(start_up.profit,start_pred))
rmse
#rmse=8854.76

r_sq_comparision=pd.DataFrame(data=['0.951,insignificant','0.962,insignificant','0.963,insignificant','0.963,insignificant','0.963,insignificant','0.96,significant'],index=['all_data','removing_influence','removing_influence','transformation','remove_marketing','remove_admin'])
r_sq_comparision
