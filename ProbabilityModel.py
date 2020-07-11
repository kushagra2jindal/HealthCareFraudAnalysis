#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:35:21 2020

@author: kushagra
"""


import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
model_GBN = GaussianNB()



TrainData = pd.read_csv("Data/Ready2/Train_ProviderWithPatientDetailsdata.csv")

TrainData = TrainData.iloc[: , 1:]

TrainData.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
TrainData.PotentialFraud=TrainData.PotentialFraud.astype('int64')


cols1 = TrainData.select_dtypes([np.number]).columns
TrainData[cols1] = TrainData[cols1].fillna(value=0)


X = TrainData.iloc[:,1:]
Y = TrainData.iloc[:,[0]]


'''regr = linear_model.LinearRegression()
regr.fit(X, Y)
'''
'''clf = DecisionTreeClassifier()
clf = clf.fit(X,Y)
'''

model_GBN.fit(X, Y)




InscClaimAmtReimbursed = 9000

AdmitForDays = 8

Gender = 1
                
RenalDiseaseIndicator = 0

ChronicCond_Alzheimer = 0

ChronicCond_Heartfailure = 0

ChronicCond_KidneyDisease = 0

ChronicCond_Cancer = 1

ChronicCond_ObstrPulmonary = 0

ChronicCond_Depression = 0

ChronicCond_Diabetes = 0

ChronicCond_IschemicHeart = 0

ChronicCond_Osteoporasis = 0

ChronicCond_rheumatoidarthritis = 0

ChronicCond_stroke = 0

Age = 81

WhetherDead = 0



if(Age <= 0 or Age>=100):
    print("Illegal data")
else:
    probability = model_GBN.predict_proba([[InscClaimAmtReimbursed,AdmitForDays,Gender,RenalDiseaseIndicator,ChronicCond_Alzheimer,ChronicCond_Heartfailure,ChronicCond_KidneyDisease,ChronicCond_Cancer,ChronicCond_ObstrPulmonary,ChronicCond_Depression,ChronicCond_Diabetes,ChronicCond_IschemicHeart,ChronicCond_Osteoporasis,ChronicCond_rheumatoidarthritis,ChronicCond_stroke,Age,WhetherDead]])
    prob = probability[0][1]
    print(prob)
    if prob >= 0.5:
        print ("No Potential Fraud Detected")
    else:
        print ("Potential Fraud Detected")














