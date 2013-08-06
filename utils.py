__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

'''
Generic utility functions useful in data-mining projects
'''

import json
import csv
import gc

def load_data_json(filePath):
    #import JSON data into a dict
    return [json.loads(line) for line in open(filePath)]

def data_garbage_collection(frmTrn,frmTest,frmAll):
    # Clean up unused frames:
    frmTrn[0] = '';frmTrn[2] = '';
    frmTest[0] = '';frmTest[2] = '';
    frmAll[1] = ''

    #garbage collection on memory
    gc.collect();
    return frmTrn,frmTest,frmAll
