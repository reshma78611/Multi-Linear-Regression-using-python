# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:26:28 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mileage=pd.read_csv('C:/Users/HP/Desktop/datasets/Cars.csv')


################EDA#################
import seaborn as sns
sns.pairplot(mileage)
# scatter plot of all data
mileage.corr()
#correlation 


##########Model building###########
import statsmodels.formula.api as smf
model=smf.ols('MPG~HP+VOL+SP+WT',data=mileage).fit()
model.params
model.summary()
#R_sq=0.77
#we observe collinearity between  VOl,WT since p-val > 0.05
#Vol and WT are insignificant

model_vol=smf.ols('MPG~VOL',data=mileage).fit()
model_vol.params
model_vol.summary()
# with vol alone it is significant

model_wt=smf.ols('MPG~WT',data=mileage).fit()
model_wt.params
model_wt.summary()
# with wt alone it is significant

model_vol_wt=smf.ols('MPG~VOL+WT',data=mileage).fit()
model_vol_wt.params
model_vol_wt.summary()
# with vol and wt, both are not significant, so we need to eliminate one of them what to eliminate is decided based on VIF


###################Validation techniques################
#check for collinearity using VIF(Variance inflation factor)
rsq_hp=smf.ols('HP~VOL+SP+WT',data=mileage).fit().rsquared
VIF_hp=1/(1-rsq_hp)

rsq_vol=smf.ols('VOL~HP+SP+WT',data=mileage).fit().rsquared
VIF_vol=1/(1-rsq_vol)

rsq_sp=smf.ols('SP~HP+VOL+WT',data=mileage).fit().rsquared
VIF_sp=1/(1-rsq_sp)

rsq_wt=smf.ols('WT~HP+VOL+SP',data=mileage).fit().rsquared
VIF_wt=1/(1-rsq_wt)

d1={'varaibles':['HP','VOL','SP','WT'],'VIF':[VIF_hp,VIF_vol,VIF_sp,VIF_wt]}
VIF_frame=pd.DataFrame(d1)
VIF_frame
#if vif>20 then collinearity
# here we observe  high collinearity b/w vol and wt
# now we need to build a model by excluding  WT as its VIF value is high


####################Validation plots##################

import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model)
# added varible plot for weight is not showing any significance 

#######################Deletion Diagnostic#################
sm.graphics.influence_plot(model)
#to find influencers or outliers so that we can delete them
#here 76th data points are outlier
# so remove 76th data point and also apply transformation (from line 53) and go for same process till now


#############################  Iteration 1  #######################

mileage1=mileage.drop(mileage.index[76],axis=0)
#76th  observation is deleted, now build new model with this data
model1=smf.ols('MPG~HP+SP+VOL+WT',data=mileage1).fit()
model1.params
model1.summary()
#R_sq=81.9
#vol and WT are insignificant
# from VIF values we know that we can delete WT

sm.graphics.plot_partregress_grid(model1)

sm.graphics.influence_plot(model1)
#78

#iteration 2
mileage2=mileage.drop(mileage.index[[76,78]],axis=0)
model2=smf.ols('MPG~HP+VOL+SP+WT',data=mileage2).fit()
model2.params
model2.summary()
#R_sq=84.8
#vol and WT are insignificant
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)
#65

#iteration 3,4,5
#transformation
mileage['HP_sq']=mileage.HP*mileage.HP
mileage['SP_sq']=mileage.SP*mileage.SP
mileage3=mileage.drop(mileage.index[[76,78,65,79]],axis=0)
model3=smf.ols('MPG~HP+VOL+SP',data=mileage3).fit()
#to make significant deleted WT
model3.params
model3.summary()
#R_sq=84.8,86.1,88.7
sm.graphics.plot_partregress_grid(model3)
sm.graphics.influence_plot(model3)
#65,79,70

#Final iteration
mileage4=mileage.drop(mileage.index[[76,78,65,79,70]],axis=0)
final_model=smf.ols('MPG~HP+SP+VOL',data=mileage4).fit()
final_model.params
final_model.summary()
#R_sq=88.8
final_pred=final_model.predict(mileage4)
final_pred

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(mileage4.MPG,final_pred))
rmse
#rmse=3.022