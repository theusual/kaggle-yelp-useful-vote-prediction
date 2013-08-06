__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'
'''
Calculate and extract features for ML
-This module is focused on calc and extraction of features for use in ML
'''

from datetime import datetime
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.externals import joblib

def cross_validate(mtxTrn,mtxTarget,clf,folds=5,SEED=42,test_size=.15):
    mean_rmse = 0.0
    #scores = cross_validation.cross_val_score(clf, mtxTrn, mtxTarget, cv=folds, random_state=i*SEED+1, test_size=test_size, scoring=)
    for i in range(folds):
        #For each fold, create a test set (test_holdout) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=i*SEED+1)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        #Train model and make predictions on CV set
        clf.fit(train_cv, y_target)
        preds = clf.predict(test_cv)

        #For this CV fold, measure the error (distance between the predictions and the actual targets)
        rmse = metrics.mean_squared_error(y_true, preds)**(.5)
        print "RMSE (fold %d/%d): %f" % (i + 1, folds, rmse)
        mean_rmse += rmse
    print "Mean RMSE: %f" % (mean_rmse/folds)

def predict(mtxTrn,mtxTarget,mtxTest,frmTest,clf,clfname):
    #fit the classifier
    clf.fit(mtxTrn, mtxTarget)

    #make predictions on test data and store them in the test data frame
    frmTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]
    return frmTest

def save_predictions(frmTest,clfname):
    timestamp = datetime.now().strftime("%d-%m-%y_%H%M")
    filename = 'Submissions/'+timestamp+'--'+clfname+'.csv'

    #-perform any manual predictions cleanup that may be necessary-#
    ##convert any predictions below 1 to 1's
    frmTest['predictions_'+clfname] = [x if x > 1 else 1 for x in frmTest['predictions_'+clfname]]

    #save predictions
    frmTest.ix[:,['user_id','business_id','predictions_'+clfname]].to_csv(filename,cols=['user_id','business_id','stars'], index=False)
    print 'Submission file saved as ',filename

def save_model(clf,clfname):
    timestamp = datetime.now().strftime("%d-%m-%y_%H%M")
    filename = 'Models/'+timestamp+'--'+clfname+'.joblib.pk1'
    joblib.dump(clf, filename, compress=9)
    print 'Model saved as ',filename

def load_model(filename):
    return joblib.load(filename)

