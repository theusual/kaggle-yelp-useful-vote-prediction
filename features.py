__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'
'''
Calculate and extract features for ML
-This module is focused on calc and extraction of features for use in ML
'''

from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np
import pandas as pd
from datetime import datetime

def handcraft(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser):
    #----------------------------------------------------------
    #Add hand-crafted features
    #----------------------------------------------------------

    ##Review age in days
    trainingDateCutoff = datetime.strptime('01-19-2013', '%m-%d-%Y'); testDateCutoff = datetime.strptime('03-12-2013', '%m-%d-%Y')
    frmTrn_All['calc_rev_age'] = [(trainingDateCutoff - date).days  for date in frmTrn_All.rev_date]
    frmTest_All['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_All.rev_date]
    frmTest_NoVotes['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_NoVotes.rev_date]
    frmTest_NoUser['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_NoUser.rev_date]

    ##Avg useful votes per day
    frmTrn_All['calc_daily_avg_useful_votes'] = np.float64( frmTrn_All.rev_votes_useful[:] / np.float64(frmTrn_All.calc_rev_age[:]))

    ##Avg user useful votes per review
    frmTrn_All['calc_user_avg_useful_votes'] = np.float64( frmTrn_All.user_votes_useful[:] / np.float64(frmTrn_All.user_review_count[:]))
    frmTest_All['calc_user_avg_useful_votes'] = np.float64( frmTest_All.user_votes_useful[:] / np.float64(frmTest_All.user_review_count[:]))

    ##Review length
    frmTrn_All['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTrn_All.rev_text]
    frmTest_All['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_All.rev_text]
    frmTest_NoVotes['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_NoVotes.rev_text]
    frmTest_NoUser['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_NoUser.rev_text]

    ##Total checkins
    i=0;tempDict = {}
    for key in frmTrn_All.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    frmTrn_All['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in frmTest_All.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    frmTest_All['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in frmTest_NoVotes.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    frmTest_NoVotes['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in frmTest_NoUser.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    frmTest_NoUser['calc_total_checkins'] = pd.Series(tempDict)
    del tempDict

    #Remove data fields no longer needed
    del frmTrn_All['bus_latitude'];del frmTest_All['bus_latitude']
    del frmTrn_All['bus_longitude'];del frmTest_All['bus_longitude']
    del frmTrn_All['rev_votes_cool'];del frmTrn_All['rev_votes_funny']
    del frmTrn_All['rev_date'];del frmTest_All['rev_date']
    del frmTrn_All['chk_checkin_info'];del frmTest_All['chk_checkin_info']
    del frmTrn_All['user_votes_useful'];del frmTest_All['user_votes_useful']

    return frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser  #return our beautiful homemade features

def vectorize(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser, feature):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Vectorize a categorical feature using the DictVectorizer -- first fit it on both train and test sets to get all possible categories, then use it to transform each set into vectors
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vec = DictVectorizer().fit([{feature:value} for value in frmTrn_All.ix[:,feature].values])
    vecTrn = vec.transform([{feature:value} for value in frmTrn_All.ix[:,feature].values])
    vecTest = vec.transform([{feature:value} for value in frmTest_All.ix[:,feature].values])
    vecTest_NoVotes = vec.transform([{feature:value} for value in frmTest_NoVotes.ix[:,feature].values])
    vecTest_NoUsers = vec.transform([{feature:value} for value in frmTest_NoUser.ix[:,feature].values])

    return vecTrn, vecTest, vecTest_NoVotes, vecTest_NoUsers

def vectorize_buscategory(frmTrn_All, frmTest_All, frmTest_NoVotes, frmTest_NoUser):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    #Vectorize business category into a hand made matrix (DictVectorizer does not work because the categories are nested and there can be many per record)
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    ##bus_categories -- create binary matrix
    listCats = []
    frmTrn_All['bus_cat_1'] = ''
    frmTrn_Cats = frmTrn_All.ix[:,['bus_categories']]
    frmTest_Cats = frmTest_All.ix[:,['bus_categories']]
    frmTest_NoVotes_Cats = frmTest_NoVotes.ix[:,['bus_categories']]
    frmTest_NoUser_Cats = frmTest_NoUser.ix[:,['bus_categories']]
    j=0
    #make a complete list of all categories in the test set by extracting them from the nested lists
    for row in frmTest_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                listCats.append(i)
        j+=1
    #Take the top 75 categories
    indexer = pd.Series(listCats).value_counts()[:75]
    #create new dataframes with one column for each category and initialize the values to 0
    for row in indexer.index.tolist():
        frmTrn_Cats[row] = 0
        frmTest_Cats[row] = 0
        frmTest_NoVotes_Cats[row] = 0
        frmTest_NoUser_Cats[row] = 0
    del frmTrn_All['bus_cat_1'];del frmTrn_Cats['bus_categories'];del frmTest_Cats['bus_categories']
    #Iterate through every record in each data set and if the category matches any of the columns, then set value in category data frame to 1
    j=0
    for row in frmTrn_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in indexer.index.tolist():
                    frmTrn_Cats[i][j] = 1
        j+=1
    j=0
    for row in frmTest_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in indexer.index.tolist():
                    frmTest_Cats[i][j] = 1
        j+=1
    j=0
    for row in frmTest_NoVotes.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in indexer.index.tolist():
                    frmTest_NoVotes_Cats[i][j] = 1
        j+=1
    j=0
    for row in frmTest_NoUser.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in indexer.index.tolist():
                    frmTest_NoUser_Cats[i][j] = 1
        j+=1
    #convert dataframes into a matrix
    vecTrn_Cats = frmTrn_Cats.as_matrix()
    vecTest_All_Cats = frmTest_Cats.as_matrix()
    vecTest_NoVotes_Cats =  frmTest_NoVotes_Cats.as_matrix()
    vecTest_NoUser_Cats  =  frmTest_NoUser_Cats.as_matrix()

    return vecTrn_Cats, vecTest_All_Cats, vecTest_NoVotes_Cats, vecTest_NoUser_Cats, indexer  #return our playful cats

def standardize(frmTrn,frmTest,quant_features):
    #---------------------------------------------------------------------
    #Standardize list of quant features (remove mean and scale to unit variance)
    #---------------------------------------------------------------------
    scaler = preprocessing.StandardScaler()

    mtxTrn= scaler.fit_transform(frmTrn.ix[:,quant_features].as_matrix())
    mtxTest = scaler.transform(frmTest.ix[:,quant_features].as_matrix())

    return mtxTrn, mtxTest #standard function return (see what I did there?)