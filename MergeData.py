#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:21:03 2020

@author: kushagra
"""


import pandas as pd
import numpy as np


Train = pd.read_csv("Data/datasets_188596_421248_Train-1542865627584.csv")
Train_Beneficiarydata=pd.read_csv("Data/Train_Beneficiarydata-1542865627584.csv")
Train_Inpatientdata=pd.read_csv("Data/Train_Inpatientdata-1542865627584.csv")
Train_Outpatientdata=pd.read_csv("Data/Train_Outpatientdata-1542865627584.csv")

Test=pd.read_csv("Data/datasets_188596_421248_Test-1542969243754.csv")
Test_Beneficiarydata=pd.read_csv("Data/Test_Beneficiarydata-1542969243754.csv")
Test_Inpatientdata=pd.read_csv("Data/Test_Inpatientdata-1542969243754.csv")
Test_Outpatientdata=pd.read_csv("Data/Test_Outpatientdata-1542969243754.csv")


Train_Beneficiarydata = Train_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Train_Beneficiarydata = Train_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)

Test_Beneficiarydata = Test_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Test_Beneficiarydata = Test_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)


Train_Beneficiarydata['DOB'] = pd.to_datetime(Train_Beneficiarydata['DOB'] , format = '%Y-%m-%d')
Train_Beneficiarydata['DOD'] = pd.to_datetime(Train_Beneficiarydata['DOD'],format = '%Y-%m-%d',errors='ignore')
Train_Beneficiarydata['Age'] = round(((Train_Beneficiarydata['DOD'] - Train_Beneficiarydata['DOB']).dt.days)/365)


Test_Beneficiarydata['DOB'] = pd.to_datetime(Test_Beneficiarydata['DOB'] , format = '%Y-%m-%d')
Test_Beneficiarydata['DOD'] = pd.to_datetime(Test_Beneficiarydata['DOD'],format = '%Y-%m-%d',errors='ignore')
Test_Beneficiarydata['Age'] = round(((Test_Beneficiarydata['DOD'] - Test_Beneficiarydata['DOB']).dt.days)/365)

Train_Beneficiarydata.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Train_Beneficiarydata['DOB']).dt.days)/365),
                                 inplace=True)


Test_Beneficiarydata.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Test_Beneficiarydata['DOB']).dt.days)/365),
                                 inplace=True)


Train_Beneficiarydata.loc[Train_Beneficiarydata.DOD.isna(),'WhetherDead']=0
Train_Beneficiarydata.loc[Train_Beneficiarydata.DOD.notna(),'WhetherDead']=1


Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.isna(),'WhetherDead']=0
Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.notna(),'WhetherDead']=1


Train_Inpatientdata['AdmissionDt'] = pd.to_datetime(Train_Inpatientdata['AdmissionDt'] , format = '%Y-%m-%d')
Train_Inpatientdata['DischargeDt'] = pd.to_datetime(Train_Inpatientdata['DischargeDt'],format = '%Y-%m-%d')
Train_Inpatientdata['AdmitForDays'] = ((Train_Inpatientdata['DischargeDt'] - Train_Inpatientdata['AdmissionDt']).dt.days)+1


Test_Inpatientdata['AdmissionDt'] = pd.to_datetime(Test_Inpatientdata['AdmissionDt'] , format = '%Y-%m-%d')
Test_Inpatientdata['DischargeDt'] = pd.to_datetime(Test_Inpatientdata['DischargeDt'],format = '%Y-%m-%d')
Test_Inpatientdata['AdmitForDays'] = ((Test_Inpatientdata['DischargeDt'] - Test_Inpatientdata['AdmissionDt']).dt.days)+1


Train_Allpatientdata=pd.merge(Train_Outpatientdata,Train_Inpatientdata,left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider','InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician','OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2','ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5','ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8','ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1','ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4','ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid','ClmAdmitDiagnosisCode'],right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider','InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician','OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2','ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5','ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8','ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1','ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4','ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid','ClmAdmitDiagnosisCode'],how='outer')

Test_Allpatientdata=pd.merge(Test_Outpatientdata,Test_Inpatientdata,left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider','InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician','OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2','ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5','ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8','ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1','ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4','ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid','ClmAdmitDiagnosisCode'],right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider','InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician','OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2','ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5','ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8','ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1','ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4','ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid','ClmAdmitDiagnosisCode'],how='outer')


Train_AllPatientDetailsdata=pd.merge(Train_Allpatientdata,Train_Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')

Test_AllPatientDetailsdata=pd.merge(Test_Allpatientdata,Test_Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')


Train_ProviderWithPatientDetailsdata=pd.merge(Train,Train_AllPatientDetailsdata,on='Provider')

Test_ProviderWithPatientDetailsdata=pd.merge(Test,Test_AllPatientDetailsdata,on='Provider')


Test_ProviderWithPatientDetailsdata_copy=Test_ProviderWithPatientDetailsdata
col_merge=Test_ProviderWithPatientDetailsdata.columns
Test_ProviderWithPatientDetailsdata=pd.concat([Test_ProviderWithPatientDetailsdata,Train_ProviderWithPatientDetailsdata[col_merge]])


remove_these_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode', 'AdmissionDt',
       'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD',
        'State', 'County', 'Provider', 'DeductibleAmtPaid', 'Race', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']


Train_category_removed = Train_ProviderWithPatientDetailsdata.drop(axis=1,columns=remove_these_columns)
Test_category_removed = Test_ProviderWithPatientDetailsdata.drop(axis=1,columns=remove_these_columns)




'''Train_category_removed.Gender=Train_category_removed.Gender.astype('category')
Test_category_removed.Gender=Test_category_removed.Gender.astype('category')

Train_category_removed.Race=Train_category_removed.Race.astype('category')
Test_category_removed.Race=Test_category_removed.Race.astype('category')

Train_category_removed=pd.get_dummies(Train_category_removed,columns=['Gender','Race'],drop_first=True)
Test_category_removed=pd.get_dummies(Test_category_removed,columns=['Gender','Race'],drop_first=True)



Test_category_removed.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
Train_category_removed.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)'''





Train_category_removed.to_csv('Data/Ready2/Train_ProviderWithPatientDetailsdata.csv')

Test_category_removed.to_csv('Data/Ready2/Test_ProviderWithPatientDetailsdata.csv')



















