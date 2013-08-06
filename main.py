__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

import utils
import munge
import features
import train

import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from scipy.sparse import coo_matrix, hstack, vstack


###def main():
#----------------------------------------#
#-------Data Loading/Cleaning/Munging----#
#----------------------------------------#

#--load training and test data frames--#
frmTrn, frmTest = munge.load_data_frames()

#--load sent score data frames--#
frmTrnSent, frmTestSent = munge.load_sent_score()

#--Data Cleaning--#
frmTrn,frmTest = munge.data_cleaning(frmTrn,frmTest)

#--Data renaming--#
frmTrn,frmTest = munge.data_renaming(frmTrn,frmTest)

#--Data type compression to save on overhead--#
frmTrn,frmTest,frmTrnSent,frmTestSent = munge.data_compression(frmTrn,frmTest,frmTrnSent,frmTestSent)

#--combine training and test data for businesses, users, and checkins [1,2,3] to create comprehensive data sets.  frmAll[0] is empty.--#
frmAll = munge.load_combined_data_frames(frmTrn,frmTest)

#--Data merging--#
frmTrn_All,frmTest_All,frmTest_NoVotes,frmTest_NoUser = munge.data_merge(frmTrn,frmTest,frmAll,frmTrnSent,frmTestSent)

#----------------------------------------#
#--------- Feature Selection-------------#
#----------------------------------------#

#--Add handcrafted (calculated) features--#
frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser = features.handcraft(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser)

#--Vectorize categorical features--#
vecTrn_Zip, vecTest_All_Zip, vecTest_NoVotes_Zip, vecTest_NoUsers_Zip = features.vectorize(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser, 'bus_zip_code')
vecTrn_BusOpen, vecTest_All_BusOpen, vecTest_NoVotes_BusOpen, vecTest_NoUsers_BusOpen = features.vectorize(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser, 'bus_open')
vecTrn_Cats, vecTest_All_Cats, vecTest_NoVotes_Cats, vecTest_NoUser_Cats, topCats = features.vectorize_buscategory(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser)

#---------Begin Model Specific Sections-------------#
##NOTE -- Not currently used features:  'calc_total_user_votes_scaled', 'user_votes_useful'

#--------------_All Model---------------------------#
#--Select quant features to be used and standardize them (remove the mean and scale to unit variance)--#
quant_features = ['rev_stars','user_average_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age','calc_user_avg_useful_votes','bus_review_count']
mtxTrn,mtxTest = features.standardize(frmTrn_All,frmTest_All,quant_features)
#Combine the standardized quant features and the vectorized categorical features
mtxTrn = hstack([mtxTrn,vecTrn_Cats,vecTrn_Zip,vecTrn_BusOpen])
mtxTest = hstack([mtxTest,vecTest_All_Cats,vecTest_All_Zip,vecTest_All_BusOpen])

#--------------_NoVotes Model---------------------------#
#--Select quant features to be used and standardize them (remove the mean and scale to unit variance)--#
quant_features =  ['rev_stars','user_average_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age']
mtxTrn,mtxTest = features.standardize(frmTrn_All,frmTest_All,quant_features)
#Combine the standardized quant features and the vectorized categorical features
mtxTrn = hstack([mtxTrn,vecTrn_Cats,vecTrn_Zip,vecTrn_BusOpen])
mtxTest = hstack([mtxTest,vecTest_All_Cats,vecTest_All_Zip,vecTest_All_BusOpen])

#--------------_NoUsers Model---------------------------#
#--Select quant features to be used and standardize them (remove the mean and scale to unit variance)--#
quant_features = ['rev_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age']
mtxTrn,mtxTest = features.standardize(frmTrn_All,frmTest_All,quant_features)
#Combine the standardized quant features and the vectorized categorical features
mtxTrn = hstack([mtxTrn,vecTrn_Cats,vecTrn_Zip,vecTrn_BusOpen])
mtxTest = hstack([mtxTest,vecTest_All_Cats,vecTest_All_Zip,vecTest_All_BusOpen])
#---------End Model Specific Sections-------------#

#--Memory cleanup prior to running the memory intensive classifiers--#
frmTrn,frmTest,frmAll = utils.data_garbage_collection(frmTrn,frmTest,frmAll)

#----------------------------------------#
#--------- Machine Learning--------------#
#----------------------------------------#
#--select target--#
mtxTarget = frmTrn_All.ix[:,['rev_votes_useful']].as_matrix()

#--select classifier--#
##  Common options:  ensemble -- RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
##                   linear_model -- SGDRegressor, Lasso
#clf = linear_model.LassoCV(cv=3)
#clf = linear_model.ElasticNet()
#clf = ensemble.RandomForestRegressor(n_estimators=50)
clf = linear_model.SGDRegressor(alpha=0.001, n_iter=1000,shuffle=True); clfname='SGD_001_1000'

#--Use classifier for cross validation--#
train.cross_validate(mtxTrn,mtxTarget,clf,folds=10,SEED=42,test_size=.15)

#--Use classifier for predictions--#
frmTest_All = train.predict(mtxTrn,mtxTarget,mtxTest,frmTest_All,clf,clfname)
#frmTest_NoVotes = train.predict(mtxTrn,mtxTarget,mtxTest,frmTest_NoVotes,clf,clfname)
#frmTest_NoUser = train.predict(mtxTrn,mtxTarget,mtxTest,frmTest_NoUser,clf,clfname)

#--Save predictions to file--#
train.save_predictions(frmTest_All,clfname)
#train.save_predictions(frmTest_NoVotes,clfname)
#train.save_predictions(frmTest_NoUser,clfname)

#--Save model to joblib file--#
train.save_model(clf,clfname)

#--Load model from joblib file--#
clf = train.load_model('Models/07-07-13_1247--SGD_001_1000.joblib.pk1')

if __name__ == '__main__':
    main()