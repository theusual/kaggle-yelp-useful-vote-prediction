__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

'''
Data loading, cleaning, and merging
-This module is focused on getting the data into a usable form for analysis (ETL/munging).
'''

import utils
import pandas as pd
import numpy as np
from datetime import datetime
import gc

def load_data_frames():
    #--------------------------------------------------------------------
    #Load the training and test data into an array of PANDAS data frames
    #--------------------------------------------------------------------
    dataDirectory = "Data/"
    dataFilesTrn = ["yelp_training_set_review.json",
                    "yelp_training_set_business.json",
                    "yelp_training_set_user.json",
                    "yelp_training_set_checkin.json"]
    dataFilesTest = ["yelp_test_set_review.json",
                    "yelp_test_set_business.json",
                    "yelp_test_set_user.json",
                    "yelp_test_set_checkin.json"]
    frmTrn = []
    frmTest = []

    for file in dataFilesTrn:
        frmTrn.append( pd.DataFrame( utils.load_data_json(dataDirectory+file) ) )
    for file in dataFilesTest:
        frmTest.append( pd.DataFrame( utils.load_data_json(dataDirectory+file) ) )
    return frmTrn, frmTest

def load_sent_score():
    #-------------------------------------------------------
    #Load pre-calculated sentiment scores for each review
    #-------------------------------------------------------
    frmTrnSent = pd.read_csv("Data/yelp_training_set_sent_score.csv",
                       names = ['sent_score','review_id'])
    frmTestSent = pd.read_csv("Data/yelp_test_set_sent_score.csv",
                        names = ['sent_score', 'review_id'])
    return frmTrnSent, frmTestSent

def data_cleaning(frmTrn,frmTest):
    #----------------------------------------------------------------------------------------------
    #Clean the data of inconsistencies, bad date fields, bad data types, nested columns, etc.
    #----------------------------------------------------------------------------------------------

    #----Convert any data types------------
    #Review Date - unicode data into datetime
    frmTrn[0].date = [datetime.strptime(date, '%Y-%m-%d') for date in frmTrn[0].date]
    frmTest[0].date = [datetime.strptime(date, '%Y-%m-%d') for date in frmTest[0].date]

    #----Flatten any nested columns--------
    #user votes
    frmTrn[2]['votes_cool'] = [rec['cool'] for rec in frmTrn[2].votes];
    frmTrn[2]['votes_funny'] = [rec['funny'] for rec in frmTrn[2].votes]
    frmTrn[2]['votes_useful'] = [rec['useful'] for rec in frmTrn[2].votes]

    #review votes
    frmTrn[0]['votes_cool'] = [rec['cool'] for rec in frmTrn[0].votes]
    frmTrn[0]['votes_funny'] = [rec['funny'] for rec in frmTrn[0].votes]
    frmTrn[0]['votes_useful'] = [rec['useful'] for rec in frmTrn[0].votes]

    #----Data extractions------------
    #Extract zip code from full address
    frmTrn[1]['zip_code'] = [str(addr[-5:]) if 'AZ' not in addr[-5:] else 'Missing' for addr in frmTrn[1].full_address]
    frmTest[1]['zip_code'] = [str(addr[-5:]) if 'AZ' not in addr[-5:] else 'Missing' for addr in frmTest[1].full_address]

    #----Round or bin any continuous variables----
    ## Note: Binning is not needed for continuous variables when using RF or linear regression models, however binning can be useful
    ## for creating categorical variables

    #Round Longitude,Latitude
    frmTrn[1]['longitude_rounded2'] = [round(x,2) for x in frmTrn[1].longitude]
    frmTrn[1]['latitude_rounded2'] = [round(x,2) for x in frmTrn[1].latitude]
    frmTest[1]['longitude_rounded2'] = [round(x,2) for x in frmTest[1].longitude]
    frmTest[1]['latitude_rounded2'] = [round(x,2) for x in frmTest[1].latitude]

    #----delete any unused and/or redundant data from the frames----
    del frmTrn[1]['type'];del frmTest[1]['type']
    del frmTrn[3]['type'];del frmTest[3]['type']
    del frmTrn[0]['type'];del frmTest[0]['type']
    del frmTrn[2]['type'];del frmTest[2]['type']
    del frmTrn[1]['full_address'];del frmTest[1]['full_address']
    del frmTrn[0]['votes']
    del frmTrn[2]['votes']
    del frmTrn[1]['neighborhoods'];del frmTest[1]['neighborhoods']
    del frmTrn[1]['state'];del frmTest[1]['state']
    del frmTrn[1]['city'];del frmTest[1]['city']
    del frmTrn[1]['name'];del frmTest[1]['name']
    del frmTest[2]['name'];del frmTrn[2]['name']
    
    return frmTrn,frmTest #---return the fresh and clean data!---

def data_renaming(frmTrn,frmTest):
    #----------------------------------------------------------------------------------------------------------------
    #Data renaming as needed (to help with merges between tables that field names overlap, etc.)
    #----------------------------------------------------------------------------------------------------------------

    #rename all columns for clarity, except the keys
    frmTrn[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrn[0]]
    frmTest[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTest[0]]
    frmTrn[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrn[1]]
    frmTest[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTest[1]]
    frmTrn[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrn[2]]
    frmTest[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTest[2]]
    frmTrn[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrn[3]]
    frmTest[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTest[3]]

    return frmTrn,frmTest  #---return the newly christened data!---

def data_compression(frmTrn,frmTest,frmTrnSent,frmTestSent):
    #------------------------------------------------------------------------------------
    #Data type compression to save on overhead
    #------------------------------------------------------------------------------------

    frmTrnSent.sent_score = frmTrnSent.sent_score.astype(np.int16)
    frmTrn[1].bus_review_count = frmTrn[1].bus_review_count.astype(np.int32)
    frmTest[1].bus_review_count = frmTest[1].bus_review_count.astype(np.int32)
    frmTrn[0].rev_stars = frmTrn[0].rev_stars.astype(np.int8)
    frmTrn[0].rev_votes_cool = frmTrn[0].rev_votes_cool.astype(np.int16)
    frmTrn[0].rev_votes_useful = frmTrn[0].rev_votes_useful.astype(np.int16)
    frmTrn[0].rev_votes_funny = frmTrn[0].rev_votes_funny.astype(np.int16)
    frmTestSent.sent_score = frmTestSent.sent_score.astype(np.int16)
    frmTest[0].rev_stars = frmTest[0].rev_stars.astype(np.int8)

    return frmTrn,frmTest,frmTrnSent,frmTestSent

def load_combined_data_frames(frmTrn,frmTest):
    #------------------------------------------------------------------------------------------------------
    #combine training and test data for businesses and checkins to create comprehensive data sets
    #------------------------------------------------------------------------------------------------------
    frmAll = ['','','','']
    for i in (1,2,3):
        frmAll[i] = frmTrn[i].append(frmTest[i])
    return frmAll

def data_merge(frmTrn,frmTest,frmAll,frmTrnSent,frmTestSent):
    #-------------------------------------------------------------------
    #Data Merging - create complete data sets for analysis and modeling
    #-------------------------------------------------------------------
    ## Create _All data sets - for the reviews that have valid references to businesses, checkins, and users WITH user votes (uses frmTrn[2], users from training data)
    frmTrn_All = frmTrn[0].merge(frmAll[1],how='inner', on='business_id')
    frmTrn_All = frmTrn_All.merge(frmTrn[2],how='inner', on='user_id')
    frmTrn_All = frmTrn_All.merge(frmAll[3],how='left', on='business_id')
    frmTrn_All = frmTrn_All.merge(frmTrnSent,how='left', on='review_id')
    frmTest_All = frmTest[0].merge(frmAll[1],how='inner', on='business_id')
    frmTest_All = frmTest_All.merge(frmTrn[2],how='inner', on='user_id')
    frmTest_All = frmTest_All.merge(frmAll[3],how='left', on='business_id')
    frmTest_All = frmTest_All.merge(frmTestSent,how='left', on='review_id')

    ## Create _NoVotes data set - for the reviews that have valid references to businesses and users from test data set (user records WITHOUT user votes)
    frmTest_NoVotes = frmTest[0].merge(frmAll[1],how='inner', on='business_id')
    frmTest_NoVotes = frmTest_NoVotes.merge(frmTest[2],how='inner', on='user_id')
    frmTest_NoVotes = frmTest_NoVotes.merge(frmAll[3],how='left', on='business_id')
    frmTest_NoVotes= frmTest_NoVotes.merge(frmTestSent,how='left', on='review_id')
    
    ## Create _NoUser data set -- for the reviews that have valid references to businesses only, user record is missing
    frmTest_NoUser= frmTest[0].merge(frmAll[1],how='inner', on='business_id')
    frmTest_NoUser = frmTest_NoUser.merge(frmAll[2],how='left', on='user_id')
    frmTest_NoUser = frmTest_NoUser.fillna(999)
    frmTest_NoUser = frmTest_NoUser[frmTest_NoUser['user_average_stars'] == 999]
    frmTest_NoUser = frmTest_NoUser.merge(frmAll[3],how='left', on='business_id')
    frmTest_NoUser = frmTest_NoUser.merge(frmTestSent,how='left', on='review_id')
    del frmTest_NoUser['user_votes_useful'];del frmTest_NoUser['user_average_stars'];del frmTest_NoUser['user_review_count']
    
    return frmTrn_All,frmTest_All,frmTest_NoVotes,frmTest_NoUser  #---return our new babies---

