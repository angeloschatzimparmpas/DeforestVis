from asyncio import gather
from itertools import count
from unittest import result
from flask import Flask, render_template, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS, cross_origin

import json
import copy
import warnings
import re
import random
import math  
import pandas as pd 
import numpy as np
import multiprocessing
from operator import itemgetter
from rulelist import RuleList
import umap
import bisect

from scipy.linalg import orthogonal_procrustes
from joblib import Memory
from joblib import Parallel, delayed

from sklearn import preprocessing
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import _tree
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import DBSCAN

# this block of code is for the connection between the server, the database, and the client (plus routing)

# access MongoDB 
app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/mydb"
mongo = PyMongo(app)

cors = CORS(app, resources={r"/data/*": {"origins": "*"}})

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/Reset', methods=["GET", "POST"])
def reset():
    global DataRawLength
    global DataResultsRaw
    global previousState
    previousState = []

    global filterActionFinal
    filterActionFinal = ''

    global keySpecInternal
    keySpecInternal = 1

    global RANDOM_SEED
    RANDOM_SEED = 42

    global keyData
    keyData = 0

    global ModelSpaceUMAP
    ModelSpaceUMAP = []

    global keepOriginalFeatures
    keepOriginalFeatures = []

    global XData
    XData = []

    global roundValueSend 
    roundValueSend = 4

    global XDataNorm
    XDataNorm = []

    global yData
    yData = []

    global XDataNoRemoval
    XDataNoRemoval = []

    global XDataNoRemovalOrig
    XDataNoRemovalOrig = []
    
    global finalResultsData
    finalResultsData = []

    global detailsParams
    detailsParams = []

    global algorithmList
    algorithmList = []

    global ClassifierIDsList
    ClassifierIDsList = ''

    global RetrieveModelsList
    RetrieveModelsList = []

    global crossValidation
    crossValidation = 3

    global parametersSelData
    parametersSelData = []

    global target_names
    target_names = []

    global keyFirstTime
    keyFirstTime = True

    global target_namesLoc
    target_namesLoc = []

    global featureCompareData
    featureCompareData = []

    global columnsKeep
    columnsKeep = []

    global columnsNewGen
    columnsNewGen = []

    global columnsNames
    columnsNames = []

    global geometry
    geometry = []

    global width
    global depth
    global gridSize

    global fileName
    fileName = []

    global results
    results = []

    global collectStatisticsPerModel
    collectStatisticsPerModel = []

    global collectDecisionsPerModel
    collectDecisionsPerModel = []

    global collectInfoPerModelPandas
    collectInfoPerModelPandas = []

    global collectLocationsAllPerSorted
    collectLocationsAllPerSorted = []

    global collectRoundingPandas
    collectRoundingPandas = []

    global metrics
    global parameters

    global scoring
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'}

    global names_labels
    names_labels = []

    global randomSearchVar
    randomSearchVar = 50

    global paramsRF
    paramsRF = {}
    
    global X_testRounded

    global paramsAB
    paramsAB = {}

    global df_cv_results_classifiers 
    df_cv_results_classifiers = pd.DataFrame()

    return 'The reset was done!'

# retrieve data from client and select the correct data set
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequest', methods=["GET", "POST"])
def retrieveFileName():
    global DataRawLength
    global DataResultsRaw
    global DataResultsRawTest
    global DataRawLengthTest

    global fileName
    fileName = []
    fileName = request.get_data().decode('utf8').replace("'", '"')
    print(fileName)
    global keySpecInternal
    keySpecInternal = 1

    global df_cv_results_classifiers 
    df_cv_results_classifiers = pd.DataFrame()

    global filterActionFinal
    filterActionFinal = ''

    global dataSpacePointsIDs
    dataSpacePointsIDs = []

    global RANDOM_SEED
    RANDOM_SEED = 42

    global keyData
    keyData = 0

    global ModelSpaceUMAP
    ModelSpaceUMAP = []

    global keepOriginalFeatures
    keepOriginalFeatures = []

    global XData
    XData = []

    global roundValueSend 
    roundValueSend = 4

    global XDataNorm
    XDataNorm = []

    global XDataNoRemoval
    XDataNoRemoval = []

    global XDataNoRemovalOrig
    XDataNoRemovalOrig = []

    global previousState
    previousState = []

    global scoring
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'}

    global yData
    yData = []

    global finalResultsData
    finalResultsData = []

    global ClassifierIDsList
    ClassifierIDsList = ''

    global algorithmList
    algorithmList = []

    global detailsParams
    detailsParams = []

    # Initializing models

    global RetrieveModelsList
    RetrieveModelsList = []

    global resultsList
    resultsList = []

    global resultsAB 

    resultsAB = []

    global crossValidation
    crossValidation = 3

    global parametersSelData
    parametersSelData = []

    global target_names
    
    target_names = []

    global keyFirstTime
    keyFirstTime = True

    global target_namesLoc
    target_namesLoc = []

    global names_labels
    names_labels = []

    global featureCompareData
    featureCompareData = []

    global columnsKeep
    columnsKeep = []

    global columnsNewGen
    columnsNewGen = []

    global columnsNames
    columnsNames = []

    global randomSearchVar
    randomSearchVar = 10

    global paramsRF
    paramsRF = {}
    
    global paramsAB
    paramsAB = {}

    global geometry
    geometry = []

    global X_testRounded

    global collectStatisticsPerModel
    collectStatisticsPerModel = []

    global collectDecisionsPerModel
    collectDecisionsPerModel = []

    global collectInfoPerModelPandas
    collectInfoPerModelPandas = []

    global collectLocationsAllPerSorted
    collectLocationsAllPerSorted = []

    global collectRoundingPandas
    collectRoundingPandas = []

    global metrics
    global parameters

    global width
    global depth
    global gridSize

    DataRawLength = -1
    DataRawLengthTest = -1
    data = json.loads(fileName)  
    if data['fileName'] == 'HeartC':
        CollectionDB = mongo.db.HeartC.find()
        target_names.append('Healthy')
        target_names.append('Diseased')
    elif data['fileName'] == 'biodegC':
        StanceTest = True
        CollectionDB = mongo.db.biodegC.find()
        CollectionDBTest = mongo.db.biodegCTest.find()
        CollectionDBExternal = mongo.db.biodegCExt.find()
        target_names.append('Non-biodegr.')
        target_names.append('Biodegr.')
    elif data['fileName'] == 'HelocC':
        CollectionDB = mongo.db.HelocC.find()
        target_names.append('Bad')
        target_names.append('Good')
    elif data['fileName'] == 'breastC':
        CollectionDB = mongo.db.breastC.find()
        target_names.append('Malignant')
        target_names.append('Benign')
    elif data['fileName'] == 'TitanicC':
        CollectionDB = mongo.db.TitanicC.find()
        target_names.append('Survived')
        target_names.append('Died')
    elif data['fileName'] == 'CreditC':
        CollectionDB = mongo.db.CreditC.find()
        target_names.append('Rejected')
        target_names.append('Accepted')
    elif data['fileName'] == 'HappinessC':
        CollectionDB = mongo.db.HappinessC.find()
        target_names.append('HS-Level-1')
        target_names.append('HS-Level-2')
        target_names.append('HS-Level-3')
    elif data['fileName'] == 'LUSCC':
        CollectionDB = mongo.db.LUSCC.find()
        target_names.append('Dead')
        target_names.append('Alive')
    elif data['fileName'] == 'LUADC':
        CollectionDB = mongo.db.LUADC.find()
        target_names.append('Dead')
        target_names.append('Alive')
    elif data['fileName'] == 'diabetesC':
        CollectionDB = mongo.db.diabetesC.find()
        target_names.append('Positive')
        target_names.append('Negative')
    elif data['fileName'] == 'ContraceptiveC':
        CollectionDB = mongo.db.ContraceptiveC.find()
        target_names.append('Short-term')
        target_names.append('Long-term')
        target_names.append('No-use')
    elif data['fileName'] == 'VehicleC':
        CollectionDB = mongo.db.VehicleC.find()
        target_names.append('Van')
        target_names.append('Car')
        target_names.append('Bus')
    elif data['fileName'] == 'WineC':
        CollectionDB = mongo.db.WineC.find()
        target_names.append('Fine')
        target_names.append('Superior')
        target_names.append('Inferior')
    else:
        CollectionDB = mongo.db.IrisC.find()
    DataResultsRaw = []
    for index, item in enumerate(CollectionDB):
        item['_id'] = str(item['_id'])
        item['InstanceID'] = index
        DataResultsRaw.append(item)
    DataRawLength = len(DataResultsRaw)

    DataResultsRawTest = []

    dataSetSelection()

    return 'Everything is okay'

def dataSetSelection():

    global AllTargets
    global target_names
    global paramsRF
    global paramsAB
    global sendHyperRF

    target_namesLoc = []

    DataResults = copy.deepcopy(DataResultsRaw)

    for dictionary in DataResultsRaw:
        for key in dictionary.keys():
            if (key.find('*') != -1):
                target = key
                continue
        continue

    DataResultsRaw.sort(key=lambda x: x[target], reverse=True)
    DataResults.sort(key=lambda x: x[target], reverse=True)

    for dictionary in DataResults:
        del dictionary['_id']
        del dictionary['InstanceID']
        del dictionary[target]

    AllTargets = [o[target] for o in DataResultsRaw]
    AllTargetsFloatValues = []

    global fileName
    data = json.loads(fileName) 

    previous = None
    Class = 0
    for i, value in enumerate(AllTargets):
        if (i == 0):
            previous = value
            if (data['fileName'] == 'IrisC'):
                target_names.append(value)
            else:
                pass
        if (value == previous):
            AllTargetsFloatValues.append(Class)
        else:
            Class = Class + 1
            if (data['fileName'] == 'IrisC'):
                target_names.append(value)
            else:
                pass
            AllTargetsFloatValues.append(Class)
            previous = value

    ArrayDataResults = pd.DataFrame.from_dict(DataResults)

    global XData, yData, RANDOM_SEED
    XData, yData = ArrayDataResults, AllTargetsFloatValues

    global keepOriginalFeatures

    for col in XData.columns:
        keepOriginalFeatures.append(col)

    warnings.simplefilter('ignore')

    executeSearch(True, True)
    
    return 'Everything is okay'

def create_global_function():
    global estimator
    location = './cachedir'
    memory = Memory(location, verbose=0)

    # calculating for all algorithms and models the performance and other results
    @memory.cache
    def estimator(n_estimators):
        # initialize model
        print('modelsCompNow!')
        n_estimators = int(n_estimators)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED)
        model.fit(X_train, predictions)
        # set in cross-validation
        #result = cross_validate(model, X_train, predictions, cv=crossValidation, scoring='accuracy')
        # result is mean of test_score
        return model.score(X_train, predictions)

def format_n(x):
    return "{0:.3f}".format(x)

def process_model(clf, name, X, y, n_splits=3):
    # Evaluate model
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=RANDOM_SEED)
    scores = cross_validate(
        clf, X, y, scoring='roc_auc', cv=ss,
        n_jobs=-1, return_estimator=True
    )

    record = dict()
    record['model_name'] = name
    record['fit_time_mean'] = format_n(np.mean(scores['fit_time']))
    record['fit_time_std'] = format_n(np.std(scores['fit_time']))
    record['test_score_mean'] = format_n(np.mean(scores['test_score']))
    record['test_score_std'] = format_n(np.std(scores['test_score']))

    return record

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def get_rules_proba(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = ""
        
        # for p in path[:-1]:
        #     if rule != "if ":
        #         rule += " and "
        #     rule += str(p)
        # rule += " then "
        # if class_names is None:
        #     rule += "response: "+str(np.round(path[-1][0][0][0],3))
        # else:
        classes = path[-1][0][0]
        l = np.argmax(classes)
        rule += f"{np.round(100.0*classes[l]/np.sum(classes),2)}"
        # rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

# Initialize every model for each algorithm
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequestSelParameters', methods=["GET", "POST"])
def executeSearch(keyFlag, keyFlagRule):

    colors = ['rgb(153, 112, 171)', 'rgb(90, 174, 97)', 'rgb(255, 255, 255)']

    global bestModel
    global decisionstatsBest
    global metrics
    global parameters
    global collectStatisticsPerModel
    global collectDecisionsPerModel
    global collectInfoPerModelPandas
    global collectLocationsAllPerSorted
    global collectRoundingPandas
    global X_testRounded
    global width
    global depth
    global gridSize
    global geometry

    if (keyFlag == False):
        metrics = pd.read_json(metrics, orient='records')
        parameters = pd.read_json(parameters, orient='records')
        collectStatisticsPerModel = pd.read_json(collectStatisticsPerModel, orient='records')
        collectDecisionsPerModel = pd.read_json(collectDecisionsPerModel, orient='records')
        collectInfoPerModelPandas = pd.read_json(collectInfoPerModelPandas, orient='records')
        collectLocationsAllPerSorted = pd.read_json(collectLocationsAllPerSorted, orient='records')
        collectRoundingPandas = pd.read_json(collectRoundingPandas, orient='records')

    if (keyFlagRule == False):
        metrics = pd.read_json(metrics, orient='records')
        parameters = pd.read_json(parameters, orient='records')
        collectStatisticsPerModel = pd.read_json(collectStatisticsPerModel, orient='records')
        collectDecisionsPerModel = pd.read_json(collectDecisionsPerModel, orient='records')
        collectInfoPerModelPandas = pd.read_json(collectInfoPerModelPandas, orient='records')
        collectLocationsAllPerSorted = pd.read_json(collectLocationsAllPerSorted, orient='records')
        collectRoundingPandas = pd.read_json(collectRoundingPandas, orient='records')

    # print(X_testRounded)
    global decisionsBest
    global estimator
    global fileName
    global df_cv_results_classifiers
    global crossValidation
    global XData
    global XDataNorm
    global yData
    global RANDOM_SEED
    global target_names
    global keepOriginalFeatures
    global sendHyperRF
    global paramsRF
    global allParametersPerformancePerModel
    allParametersPerformancePerModel = []
    global randomSearchVar
    global X_train
    global Order
    global X_test
    global y_train
    global y_test
    global resultsAB
    global roundValueSend

    featureNames = []
    for col in XData.columns:
        featureNames.append(col)

    np.random.seed(RANDOM_SEED) # seeds
    resultsAB = []

    if (keyFlag == True):
        if (keyFlagRule == True):
            x = XData.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            XDataNorm = pd.DataFrame(x_scaled)

            featureNamesLocal = []
            for col in XData.columns:
                featureNamesLocal.append(col+'_minLim')
                
            for col in XData.columns:
                featureNamesLocal.append(col+'_maxLim')

            for ind, value in enumerate(target_names):
                featureNamesLocal.append(str(ind))

            data = json.loads(fileName)  

            X_train, X_test, y_train, y_test = train_test_split(XDataNorm, yData, test_size=0.2, stratify=yData, random_state=RANDOM_SEED)

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            X_train.columns = keepOriginalFeatures
            X_test.columns = keepOriginalFeatures

            countNumberOfFeatures = 0
            for col in XData.columns:
                countNumberOfFeatures = countNumberOfFeatures + 1
            # rootSQ = int(math.sqrt(countNumberOfFeatures))
            ebm = ExplainableBoostingClassifier(random_state=RANDOM_SEED, n_jobs=-1)
            # n_splits = 3
            # record = process_model(ebm, 'ebm', XData, yData, n_splits=n_splits)
            # print(record)
            # create_global_function()
            # params = {'n_estimators': (50, 101), 'min_samples_split': (2,11), 'max_depth': (10, 101), 'min_samples_leaf': (1,6), 'max_features': (rootSQ,countNumberOfFeatures+1) }
            # bayesopt = BayesianOptimization(estimator, params, random_state=RANDOM_SEED)
            # bayesopt.maximize(init_points=15, n_iter=5, acq='ucb') # 20 and 5
            # bestParams = bayesopt.max['params']
            # estimator = RandomForestClassifier(n_estimators=int(bestParams.get('n_estimators')), max_depth=int(bestParams.get('max_depth')), min_samples_leaf=int(bestParams.get('min_samples_leaf')), max_features=int(bestParams.get('max_features')), min_samples_split=int(bestParams.get('min_samples_split')), oob_score=True, random_state=RANDOM_SEED)
            # paramsRF = {'n_estimators': [int(bestParams.get('n_estimators'))], 'max_depth': [int(bestParams.get('max_depth'))], 'min_samples_leaf': [int(bestParams.get('min_samples_leaf'))], 'max_features': [int(bestParams.get('max_features'))], 'min_samples_split': [int(bestParams.get('min_samples_split'))] }
            # resultsLocalRF = DecisionTreeComposer(XData, X_train, y_train, X_test, estimator, paramsRF, crossValidation, RANDOM_SEED, roundValueSend) 
            ebm.fit(X_train,y_train)
            global predictions
            predictions = ebm.predict(X_train)
            predictTest = ebm.predict(X_test)
            print('Train score', accuracy_score(y_train, predictions))
            print('Test score', accuracy_score(y_test, predictTest))
            base_estimatorCompute=DecisionTreeClassifier(random_state=RANDOM_SEED)
            base_estimatorCompute.fit(X_train, predictions)
            #print(base_estimatorCompute.score(X_train, predictions))
            # rules = get_rules(base_estimatorCompute, XData.columns, target_names)
            # print(rules)
            rulesNumber = get_rules(base_estimatorCompute, XData.columns, target_names)
            countRules = 0
            for everyRule in rulesNumber:
                # if 'if' in everyRule:
                #     countRules += 1
                countRules = countRules + everyRule.count('and')
            print('number of rules', countRules)
            randomS = 50
            if (countRules >= 100):
                countRules = 140
            if (int((countRules/2)+1) <= randomS):
                countRules = randomS*2
            #params = {'n_estimators': list(range(1,12)), 'base_estimator__max_depth': list(range(1,2)), 'base_estimator__splitter': ['best','random'], 'base_estimator__criterion': ['gini','entropy'], 'base_estimator__max_features': list(range(1,countNumberOfFeatures+1))}
            params = {'n_estimators': list(range(1,int((countRules/2)+1))), 'base_estimator__max_depth': list(range(1,2)), 'base_estimator__splitter': ['best'], 'base_estimator__criterion': ['gini'], 'base_estimator__max_features': ['sqrt']}
            scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'}
            #base_estim=DecisionTreeClassifier(max_depth=1)
            #base_estim=DecisionTreeClassifier(max_depth=3)
            estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state=RANDOM_SEED)
            randSear = RandomizedSearchCV(    
                estimator=estimator, param_distributions=params, n_iter=randomS, refit='accuracy', scoring=scoring,
                cv=[(slice(None), slice(None))], verbose=0, n_jobs=-1, random_state=RANDOM_SEED)
            randSear.fit(X_train, predictions)
            print(randSear.score(X_train, predictions))
            # process the results
            cv_results = []
            cv_results.append(randSear.cv_results_)
            df_cv_results = pd.DataFrame.from_dict(cv_results)

            number_of_models = []
            # number of models stored
            number_of_models = len(df_cv_results.iloc[0][0])

            # initialize results per row
            df_cv_results_per_row = []

            modelsIDs = []
            for i in range(number_of_models):
                number = i
                modelsIDs.append(str(number))
                df_cv_results_per_item = []
                for column in df_cv_results.iloc[0]:
                    df_cv_results_per_item.append(column[i])
                df_cv_results_per_row.append(df_cv_results_per_item)

            # store the results into a pandas dataframe
            df_cv_results_classifiers = pd.DataFrame(data = df_cv_results_per_row, columns= df_cv_results.columns)
            df_cv_results_classifiers = df_cv_results_classifiers.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_base_estimator__splitter',
            'param_base_estimator__max_features', 'param_base_estimator__max_depth',
            'param_base_estimator__criterion', 'split0_test_accuracy',
            'std_test_accuracy', 'rank_test_accuracy',
            'split0_test_precision_macro', 'mean_test_precision_macro',
            'std_test_precision_macro', 'rank_test_precision_macro',
            'split0_test_recall_macro', 'mean_test_recall_macro',
            'std_test_recall_macro', 'rank_test_recall_macro'])

            df_cv_results_find_best = df_cv_results_classifiers.copy()
            df_cv_results_find_best = df_cv_results_find_best.sort_values(['mean_test_accuracy','param_n_estimators'], ascending=[False, True])
            bestModel = df_cv_results_find_best.iloc[0].name + 1

            df_cv_results_classifiers = df_cv_results_classifiers.sort_values(['param_n_estimators'], ascending=[True])
            df_cv_results_classifiers = df_cv_results_classifiers.reset_index(drop=True)
            parametersPerformancePerModel = df_cv_results_classifiers.copy()
            parametersPerformancePerModel = parametersPerformancePerModel.drop(columns=['mean_test_accuracy', 'param_n_estimators'])
            
            accuracyListInit = df_cv_results_classifiers['mean_test_accuracy'].tolist()
            accuracyList = []
            for each in accuracyListInit:
                accuracyList.append(each*100)

            parametersPerformancePerModel = parametersPerformancePerModel.to_json(double_precision=15)

            parametersLocal = json.loads(parametersPerformancePerModel)['params'].copy()
            Models = []
            for index, items in enumerate(parametersLocal):
                Models.append(str(items))

            parametersLocalNew = [ parametersLocal[your_key] for your_key in Models ]
            #df_cv_results_classifiers = df_cv_results_classifiers.drop(columns=['params'])
            #print(df_cv_results_classifiers)
            # copy and filter in order to get only the metrics
            metrics = df_cv_results_classifiers.copy()
            metrics = metrics.filter(['mean_test_accuracy']) 
            parameters = df_cv_results_classifiers.copy()
            parameters = parameters.filter(['param_n_estimators']) 
            #print(metrics)
            # parametersPerformancePerModel = pd.DataFrame()
            # concat parameters and performance
            # parametersPerformancePerModel = pd.DataFrame(df_cv_results_classifiers['params'])
            #print(parametersPerformancePerModel)
            # fit and extract the probabilities
            #cls_t = randSear.best_estimator_
            #print(cls_t)
            # create_global_function()
            # bayesopt = BayesianOptimization(estimator, params, random_state=RANDOM_SEED)
            # bayesopt.maximize(init_points=100, n_iter=20, acq='ucb') # 20 and 5
            # bestParams = bayesopt.max['params']
            #cls_t = AdaBoostClassifier(n_estimators=int(bestParams.get('n_estimators')), random_state=RANDOM_SEED)
            # cls_t.fit(X_train, predictions)
            # gatherRules = []
            # for tree_idx, est in enumerate(cls_t.estimators_):
            #     rules = get_rules(est, XData.columns, target_names)
            #     #print(rules)
            #     gatherRules.append(rules)
            #print(len(gatherRules))
            # for each in gatherRules:
            #     print(each)
            #resultsLocalRF = DecisionTreeComposer(XData, X_train, y_train, X_test, cls_t, params, crossValidation, RANDOM_SEED, roundValueSend) 
            #print(cls_t.score(X_train, predictions))
            #print(len(gatherRules))
            # print(gatherRules[0])
            # print(gatherRules[1])

            collectDecisionsPerModel = pd.DataFrame()
            collectDecisions = []
            collectDecisionsMod = []
            collectLocationsAll = []
            collectStatistics = []
            collectStatisticsMod = []
            collectStatisticsPerModel = []
            collectInfoPerModel = []
            collectDecisions = []
            collectLocations = []
            collectStatistics = []
            collectRounding = []
            sumRes = 0

            featureNamesDuplicated = []
            
            for col in XData.columns:
                featureNamesDuplicated.append(col+'_minLim')
            for col in XData.columns:
                featureNamesDuplicated.append(col+'_maxLim')

            global X_trainRounded

            global keepRoundingLevel
            counterModels = 1
            gatherErrorRates = []
            listOfRules = []
            for eachModelParameters in parametersLocalNew:
                collectDecisions = []
                collectLocations = []
                collectStatistics = []
                gatherAllRounding = []
                sumRes = 0
                estimator.set_params(**eachModelParameters)
                for r in range(0, roundValueSend):
                    X_trainRounded = X_train.copy(deep=True)
                    X_trainRounded = X_trainRounded.round(r)
                    X_testRounded = X_test.copy(deep=True)
                    X_testRounded = X_testRounded.round(r)
                    estimator.fit(X_trainRounded, predictions)
                    best = estimator.score(X_trainRounded, predictions) * 100
                    gatherAllRounding.append(best)
                    if (accuracyList[counterModels-1] == best):
                        keepRoundingLevel = r + 1
                        break

                gatherErrorRates.append(estimator.estimator_errors_)    

                for tree_idx, est in enumerate(estimator.estimators_):
                    decisionPath = extractDecisionInfo(est,counterModels,tree_idx,X_trainRounded,predictions,featureNamesDuplicated,listOfRules,feature_names=featureNames,only_leaves=True)
                    collectDecisions.append(decisionPath[0])
                    collectStatistics.append(decisionPath[1])
                    sumRes = sumRes + decisionPath[2]
                    collectLocations.append(decisionPath[3])
                collectDecisionsMod.append(collectDecisions)
                collectStatisticsMod.append(collectStatistics)
                collectInfoPerModel.append(sumRes)
                collectLocationsAll.append(collectLocations)
                collectRounding.append(gatherAllRounding)
                counterModels = counterModels + 1

            collectDecisionsSorted = []
            collectStatisticsSorted = []
            collectLocationsAllSorted = []
            collectInfoPerModelPandas = pd.DataFrame(collectInfoPerModel)
            collectRoundingPandas = pd.DataFrame(collectRounding)

            for e in Models:
                el = int(e)
                for item in collectDecisionsMod[el]:
                    collectDecisionsSorted.append(item)
                for item2 in collectStatisticsMod[el]:
                    collectStatisticsSorted.append(item2)
                for item3 in collectLocationsAll[el]:
                    collectLocationsAllSorted.append(item3)

            collectDecisionsPerModel = pd.concat(collectDecisionsSorted)
            collectStatisticsPerModel = pd.concat(collectStatisticsSorted)
            collectLocationsAllPerSorted = pd.DataFrame(collectLocationsAllSorted)
            
            listOfResults = decisionPath[4]
            listOfResultsFloat = []
            for each in listOfResults:
                for coord in each:
                    pair = [float(s) for s in coord.strip().split(",")]
                    listOfResultsFloat.append(pair)
            collectStatisticsPerModel['proba'] = listOfResultsFloat

            for indC, column in enumerate(collectRoundingPandas.columns):
                for indR, row in enumerate(collectRoundingPandas[indC]):
                    if (math.isnan(collectRoundingPandas.loc[indR, indC])):
                        collectRoundingPandas.loc[indR, indC] = collectRoundingPandas.loc[indR, indC-1]

            collectDecisionsPerModel = collectDecisionsPerModel.reset_index(drop=True)
            collectStatisticsPerModel = collectStatisticsPerModel.reset_index(drop=True) 
            collectLocationsAllPerSorted = collectLocationsAllPerSorted.reset_index(drop=True) 
            collectInfoPerModelPandas = collectInfoPerModelPandas.reset_index(drop=True)

            # collectDecisionsPerModel = pd.concat([collectDecisionsPerModel, collectStatisticsPerModel['samples']], axis=1)
            # collectDecisionsPerModel = pd.concat([collectDecisionsPerModel, collectStatisticsPerModel['predicted_value']], axis=1)
            # collectDecisionsPerModel = pd.concat([collectDecisionsPerModel, collectStatisticsPerModel['impurity']], axis=1)

            def group_duplicate_index(df):
                a = df.values
                sidx = np.lexsort(a.T)
                b = a[sidx]

                m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False] ))
                idx = np.flatnonzero(m[1:] != m[:-1])
                I = df.index[sidx].tolist()       
                return [I[i:j] for i,j in zip(idx[::2],idx[1::2]+1)]

            duplicates = group_duplicate_index(collectDecisionsPerModel)
            duplicates_sorted = sorted(duplicates, key=len, reverse=True)

            duplicates_sorted_merged = []
            for i, d in enumerate(duplicates_sorted):
                if (i % 2 == 0):
                    duplicates_sorted_merged.append(duplicates_sorted[i] + duplicates_sorted[i+1])

            def compute_alpha(error):
                '''
                Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
                alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
                error: error rate from weak classifier m
                '''
                return (0.5) * np.log((1 - error) / error)

            collectStatisticsPerModel["tree"] = " "
            collectStatisticsPerModel["commonality_ranking"] = " "
            collectStatisticsPerModel["alpha"] = " "
            collectStatisticsPerModel["probamulalpha"] = " "
            collectStatisticsPerModel["purityComb"] = " "
            enumerateTree = 0
            
            for index, row in collectStatisticsPerModel.iterrows():
                if (index % 2) == 0:
                    enumerateTree += 1
                collectStatisticsPerModel['tree'].loc[index] = enumerateTree
                collectStatisticsPerModel['alpha'].loc[index] = compute_alpha(gatherErrorRates[int(row['counterModels']-1)][int(row['tree_index'])])
                for l, el in enumerate(duplicates_sorted_merged):
                    rank = l + 1
                    if (index in el):
                        collectStatisticsPerModel['commonality_ranking'].loc[index] = rank
                    else:
                        pass

            for index, row in collectStatisticsPerModel.iterrows():
                    if (index % 2) == 0:
                        keepResultTemp = collectStatisticsPerModel['alpha'].loc[index] * collectStatisticsPerModel['proba'].loc[index][0] + collectStatisticsPerModel['alpha'].loc[index+1] * collectStatisticsPerModel['proba'].loc[index+1][0]
                        keepResultImpurity = collectStatisticsPerModel['impurity'].loc[index] + collectStatisticsPerModel['impurity'].loc[index+1]
                    collectStatisticsPerModel['probamulalpha'].loc[index] = keepResultTemp
                    collectStatisticsPerModel['purityComb'].loc[index] = keepResultImpurity

            collectStatisticsPerModel = collectStatisticsPerModel.sort_values(['counterModels', 'commonality_ranking', 'tree_index', 'node'], ascending=[True, True, True, True])
            #print(collectStatisticsPerModel)
            # collectDecisionsPerModel = collectDecisionsPerModel.drop(columns=['samples', 'predicted_value', 'impurity'])
            collectDecisionsPerModel = collectDecisionsPerModel.reindex(collectStatisticsPerModel.index)
            
            collectDecisionsPerModel = collectDecisionsPerModel.reset_index(drop=True)
            collectStatisticsPerModel = collectStatisticsPerModel.reset_index(drop=True) 
            collectLocationsAllPerSorted = collectLocationsAllPerSorted.reset_index(drop=True) 
            collectInfoPerModelPandas = collectInfoPerModelPandas.reset_index(drop=True)
    
        decisionstatsBest = collectStatisticsPerModel.loc[collectStatisticsPerModel['counterModels'] == bestModel]
        decisionsBest = collectDecisionsPerModel.iloc[decisionstatsBest.index]

        storeFeatureNames = []
        storeValues = []
        storeMinMax = []
        for ind, elem in collectDecisionsPerModel.iterrows():
            for key, value in elem.items():
                if (int(value) != 2):
                    replaceKey = key.replace("_maxLim","");
                    replaceKey = replaceKey.replace("_minLim","");
                    storeFeatureNames.append(replaceKey)
                    storeValues.append(value)

        collectStatisticsPerModel['feature'] = storeFeatureNames
        collectStatisticsPerModel['valueLimit'] = storeValues

        storeFeatureNames = []
        storeRealFeatures = []
        storeValues = []
        storeMinMax = []
        for ind, elem in decisionsBest.iterrows():
            for key, value in elem.items():
                if (int(value) != 2):
                    storeRealFeatures.append(key)
                    replaceKey = key.replace("_maxLim","");
                    replaceKey = replaceKey.replace("_minLim","");
                    storeFeatureNames.append(replaceKey)
                    storeValues.append(value)


        storeCountingFrequency = []
        for item in storeFeatureNames:
            storeCountingFrequency.append(storeFeatureNames.count(item))

        decisionstatsBest['frequency'] = storeCountingFrequency
        decisionstatsBest['feature'] = storeFeatureNames
        decisionstatsBest['valueLimit'] = storeValues
        decisionstatsBest['completeSuffix'] = storeRealFeatures

        sumFeature = decisionstatsBest.groupby('feature')['probamulalpha'].sum()
        keys = sumFeature.index.values.tolist()
        values = sumFeature.tolist()

        gatherImportances = []
        for ind, elem in decisionstatsBest.iterrows():
            for i, v in enumerate(keys):
                if (v == elem['feature']):
                    gatherImportances.append(values[i])

        decisionstatsBest['importance'] = gatherImportances

        decisionstatsBest = decisionstatsBest.sort_values(['importance', 'probamulalpha', 'frequency', 'commonality_ranking', 'tree_index', 'node'], ascending=[False, False, False, True, True, True])
        decisionsBest = decisionsBest.reindex(decisionstatsBest.index)

        decisionstatsBest = decisionstatsBest.reset_index(drop=True) 
        decisionsBest = decisionsBest.reset_index(drop=True)

        decisionstatsBest["isStatusChanged"] = decisionstatsBest["feature"].shift(1, fill_value=decisionstatsBest["feature"].head(1)) != decisionstatsBest["feature"]
        
        uniqueFeat = decisionstatsBest['feature'].unique().tolist()
        listOfFeat = [0] * len(uniqueFeat)

        width = len(uniqueFeat)
        depth = int(max(storeCountingFrequency) / 2)
        size = math.ceil(math.sqrt(len(X_trainRounded))) + 1
        gridSize = size * size

        geometry = []
        geometry.append(width)
        geometry.append(depth)
        geometry.append(size)
        geometry.append(gridSize)

        gatherPosition = []
        counter = -1
        howManyTimesInside = 0
        countPos = 0

        gatherGridPositions = []

        for ind, elem in decisionstatsBest.iterrows():
            indexPos = uniqueFeat.index(elem['feature'])

            if (ind % 2 == 0):
                if (elem["isStatusChanged"]):
                    howManyTimesInside += 1
                    counter = depth * howManyTimesInside
                    countPos += 1
                else:
                    counter += 1
                    if (ind != 0):
                        listOfFeat[indexPos] = listOfFeat[indexPos] + 1
            gatherGridPositions.append(countPos + listOfFeat[indexPos]*(len(uniqueFeat)))
            gatherPosition.append(counter)

        decisionstatsBest["position"] = gatherPosition
        decisionstatsBest["gridpos"] = gatherGridPositions 

    #print(keepRoundingLevel)

    if (keyFlag == False):
        for loop, item in decisionstatsBest.iterrows():
            if (item['gridpos'] == checkingRule):
                decisionstatsBest.loc[loop,'valueLimit'] = round(thresholdRule,keepRoundingLevel)

    storeNode1Frequency = []
    storeNode2Frequency = []
    storeGT = []
    storeWhereItBelongs = []
    for ind, elem in X_trainRounded.iterrows():
        countNode1 = 0
        countNode2 = 0
        gatherTruth = []
        gatherWhereItBelongs = []
        for key, value in elem.iteritems():
            for index, row in decisionsBest[key+'_maxLim'].items():
                if (int(row) != 2):
                    if (value <= row):
                        countNode1 += 1
                        if (collectStatisticsPerModel.loc[index, 'predicted_value'] == y_train[ind]):
                            gatherTruth.append(1)
                        else:
                            gatherTruth.append(0)
                        gatherWhereItBelongs.append(index)
            for index, row in decisionsBest[key+'_minLim'].items():
                if (int(row) != 2):
                    if (value >= row):
                        countNode2 += 1
                        if (collectStatisticsPerModel.loc[index, 'predicted_value'] == y_train[ind]):
                            gatherTruth.append(1) # correct
                        else:
                            gatherTruth.append(0) # false
                        gatherWhereItBelongs.append(index)
        storeNode1Frequency.append(countNode1)
        storeNode2Frequency.append(countNode2)
        storeGT.append(gatherTruth)
        storeWhereItBelongs.append(gatherWhereItBelongs)
    # print(storeNode1Frequency)
    # print(storeNode2Frequency)
    # print(storeWhereItBelongs)
    # print(storeGT)
    # print(len(storeNode1Frequency))
    
    buildGrid = pd.DataFrame()

    perFeatureUnique = []
    gatherUnique = []
    orderedFeatures = []
    for ind, elem in decisionstatsBest.iterrows():
        if(elem['feature'] not in orderedFeatures):
            orderedFeatures.append(elem['feature'])
        if (elem['isStatusChanged'] == False):
            if(round(elem['valueLimit'],keepRoundingLevel) not in gatherUnique):
                gatherUnique.append(round(elem['valueLimit'],keepRoundingLevel))
        else:
            gatherUnique.append(0)
            gatherUnique.append(1)
            gatherUnique.sort()
            perFeatureUnique.append(gatherUnique)
            gatherUnique = []
            if(round(elem['valueLimit'],keepRoundingLevel) not in gatherUnique):
                gatherUnique.append(round(elem['valueLimit'],keepRoundingLevel))

    gatherUnique.append(0)
    gatherUnique.append(1)
    gatherUnique.sort()
    perFeatureUnique.append(gatherUnique)
    # print(perFeatureUnique)
    # print(orderedFeatures)

    keepMaxValueList = []
    # keepMaxValueIndexList = []
    for i, each in enumerate(orderedFeatures):
        keepMaxValue = []
        # keepMaxValueIndex = []
        for j in range(0,len(perFeatureUnique[i])-1):
            sumPerClass = [0] * len(target_names)
            decisionstatsBestFiltered = decisionstatsBest[decisionstatsBest['feature'] == each]
            # divider = len(decisionstatsBestFiltered.index) / 2
            for ind, item in decisionstatsBestFiltered.iterrows():
                if ('_maxLim' in item['completeSuffix']):
                    minVal = 0
                    maxVal = round(item['valueLimit'],keepRoundingLevel)
                else:
                    minVal = round(item['valueLimit'],keepRoundingLevel)
                    maxVal = 1
                if ((perFeatureUnique[i][j] >= minVal) and (perFeatureUnique[i][j+1]) <= maxVal):
                    if (int(item['predicted_value']) == 0):
                        sumPerClass[0] = sumPerClass[0] + (item['proba'][0] * item['alpha'])
                        sumPerClass[1] = sumPerClass[1] + ((100-item['proba'][0]) * item['alpha'])
                    else:
                        sumPerClass[1] = sumPerClass[1] + (item['proba'][0] * item['alpha'])
                        sumPerClass[0] = sumPerClass[0] + ((100-item['proba'][0]) * item['alpha'])
            # max_value = max(sumPerClass)
            # max_index = sumPerClass.index(max_value)
            keepMaxValue.append(sumPerClass)
            # keepMaxValueIndex.append(max_index)
        #keepMaxValueUpdated = [item / divider for item in keepMaxValue]
        keepMaxValueList.append(keepMaxValue)   
        # keepMaxValueIndexList.append(keepMaxValueIndex) 

    resultsAnalysis = pd.DataFrame()
    for loop, element in X_testRounded.iterrows():
        for i, each in enumerate(orderedFeatures):
            for j in range(0,len(perFeatureUnique[i])-1):
                if ((element[each] >= perFeatureUnique[i][j]) and (element[each] <= perFeatureUnique[i][j+1])):
                    resultsAnalysis = resultsAnalysis.append({'ID': loop+1, each: keepMaxValueList[i][j], 'GT': y_test[loop]}, ignore_index=True)

    resultsAnalysis = resultsAnalysis.groupby('ID').first().reset_index(drop=True)

    storeAllsums = []
    storeContradict = []
    for loop, element in resultsAnalysis.iterrows():
        sumPerClass = [0] * len(target_names)
        for i, each in enumerate(orderedFeatures):
            maximumVal = max(resultsAnalysis.loc[resultsAnalysis.index[loop], each])
            minimumVal = min(resultsAnalysis.loc[resultsAnalysis.index[loop], each])
            resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_exact'] = maximumVal
            resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_diff'] = maximumVal - minimumVal
            for l, cl in enumerate(sumPerClass):
                sumPerClass[l] = sumPerClass[l] + element[each][l]
        max_value = max(sumPerClass)
        max_index = sumPerClass.index(max_value)
        storeAllsums.append(sumPerClass)
        resultsAnalysis.loc[resultsAnalysis.index[loop], 'prediction'] = max_index
        resultsAnalysis.loc[resultsAnalysis.index[loop], 'ID'] = loop
        resultsAnalysis.loc[resultsAnalysis.index[loop], 'difference'] = abs(sumPerClass[1] - sumPerClass[0])

    for loop, element in resultsAnalysis.iterrows():
        if (element['prediction'] == element['GT']):
            storeContradict.append(1) # correct
        else:
            storeContradict.append(0) # wrong

    resultsAnalysis['total'] = storeAllsums
    resultsAnalysis['noncontradiction'] = storeContradict
    resultsAnalysis = resultsAnalysis.sort_values(['noncontradiction', 'difference'], ascending=[True, True])
    resultsAnalysis = resultsAnalysis.reset_index(drop=True)

    countW = resultsAnalysis['noncontradiction'].value_counts()[0]
    countC = resultsAnalysis['noncontradiction'].value_counts()[1]
    print('Count wrong : ', countW)
    print('Count correct : ', countC)
    for pos in range(0,width*depth):
        tempDFNode1 = pd.DataFrame(columns = ['ID' , 'changeFrequency', 'colorGT' , 'rawGT'])
        tempDFNode12 = pd.DataFrame(columns = ['ID' , 'changeFrequency', 'colorGT' , 'rawGT'])
        tempDFNode2 = pd.DataFrame(columns = ['ID' , 'changeFrequency', 'colorGT' , 'rawGT'])
        resultingDF = pd.DataFrame()
        col_pos_list = decisionstatsBest['position'].tolist()
        if (pos in col_pos_list):
            for ind, elem in decisionstatsBest.iterrows():
                if (elem['position'] == pos):
                    if (elem['node'] == 1):
                        for i, item in X_trainRounded.iterrows():
                            if (item[elem['feature']] <= elem['valueLimit']):
                                tempDFNode1 = tempDFNode1.append({'ID' : i,
                                    'changeFrequency' : storeNode1Frequency[i],
                                    'colorGT': colors[y_train[i]],
                                    'rawGT': y_train[i]} , 
                                    ignore_index=True)
                        tempDFNode1 = tempDFNode1.sort_values(['rawGT', 'changeFrequency'], ascending=[False, True])
                        for middle in range(0,(gridSize - len(X_trainRounded))):
                            tempDFNode12 = tempDFNode12.append({'ID' : -1,
                                'changeFrequency' : 0,
                                'colorGT': colors[2],
                                'rawGT': 3} , 
                                ignore_index=True)
                    else:
                        for i, item in X_trainRounded.iterrows():
                            if (item[elem['feature']] >= elem['valueLimit']):
                                tempDFNode2 = tempDFNode2.append({'ID' : i,
                                    'changeFrequency' : storeNode2Frequency[i],
                                    'colorGT': colors[y_train[i]],
                                    'rawGT': y_train[i]} , 
                                    ignore_index=True)
                        tempDFNode2 = tempDFNode2.sort_values(['rawGT', 'changeFrequency'], ascending=[False, False])
            resultingDF = pd.concat([tempDFNode1,tempDFNode12,tempDFNode2])
            buildGrid = pd.concat([buildGrid,resultingDF])
        else:
            for middle in range(0,gridSize):
                tempDFNode12 = tempDFNode12.append({'ID' : -1,
                    'changeFrequency' : 0,
                    'colorGT': colors[2],
                    'rawGT': 3} , 
                    ignore_index=True)
            buildGrid = pd.concat([buildGrid,tempDFNode12])

    buildGrid = buildGrid.reset_index(drop=True)
    buildGrid['value'] = buildGrid.index

    print(decisionstatsBest)

    probabilitiesGen = []
    colorsGen = []
    colorsSplitGen = []
    alphaGen = []
    splitGen = []
    for i in range(0,geometry[1]):
        for j in range(0,len(orderedFeatures)):
            probabilities = []
            colorEach1 = []
            probabilities2 = []
            colorEach2 = []
            probabilitiesIns = []
            colorEachIns = []
            alphaIns = []
            splitIns = []
            colorsSplitIns = []
            item = (i * len(orderedFeatures)) + j
            for ind, el in decisionstatsBest.iterrows():
                if (item == el['gridpos']):
                    if ('_minLim' in el['completeSuffix']):
                        alphaIns.append(el['alpha'])
                        splitIns.append(el['valueLimit'])
                    else:
                        splitIns.append(1 - el['valueLimit'])
                    if (el['predicted_value'] == 1):
                        # if (el['gridpos'] == 2):
                        #     colorsSplitIns.append('rgb(90, 174, 97)')
                        # else:
                        colorsSplitIns.append('rgb(153, 112, 171)')
                    else:
                        # if (el['gridpos'] == 2):
                        #     colorsSplitIns.append('rgb(90, 174, 97)')
                        # else:
                        colorsSplitIns.append('rgb(90, 174, 97)')
                    probabilities.append(el['proba'][0])
                    probabilities2.append(100-el['proba'][0])
                    if (el['predicted_value'] == 0):
                        colorEach1.append('rgb(153, 112, 171)')
                        colorEach2.append('rgb(90, 174, 97)')
                    else:
                        colorEach1.append('rgb(90, 174, 97)')
                        colorEach2.append('rgb(153, 112, 171)')
            probabilitiesIns.append(probabilities)
            probabilitiesIns.append(probabilities2)
            probabilitiesGen.append(probabilitiesIns)
            alphaGen.append(alphaIns)
            splitInsRev = list(reversed(splitIns))
            splitGen.append(splitInsRev)
            colorsSplitInsRev = list(reversed(colorsSplitIns))
            colorsSplitGen.append(colorsSplitInsRev)
            colorEachIns.append(colorEach1)
            colorEachIns.append(colorEach2)
            colorsGen.append(colorEachIns)

    resultsAnalysisConfused = resultsAnalysis.loc[resultsAnalysis['GT'] != resultsAnalysis['prediction']].sort_values(['difference'], ascending=[False])
    resultsAnalysisCorrect = resultsAnalysis.loc[resultsAnalysis['GT'] == resultsAnalysis['prediction']]

    resultsAnalysis = pd.DataFrame()
    resultsAnalysis = pd.concat([resultsAnalysisConfused,resultsAnalysisCorrect])
    resultsAnalysis = resultsAnalysis.reset_index(drop=True)

    TestIDs = resultsAnalysis['ID'].tolist()
    GTTest = resultsAnalysis['GT'].tolist()
    PredTest = resultsAnalysis['prediction'].tolist()
    diffTest = resultsAnalysis['difference'].tolist()
    index_list = resultsAnalysis.index.values.tolist()

    GTTestCol = []
    PredTestCol = []
    rectWidths = []
    rectWidthsRemaining = []
    for i in index_list:
        rectWidths.append(4)
        rectWidthsRemaining.append(0.5)
        if (GTTest[i] == 1):
            GTTestCol.append('rgb(90, 174, 97)')
        else:
            GTTestCol.append('rgb(153, 112, 171)')
        if (PredTest[i] == 1):
            PredTestCol.append('rgb(90, 174, 97)')
        else:
            PredTestCol.append('rgb(153, 112, 171)')

    
    for loop, element in resultsAnalysis.iterrows():
        findSum = 0
        for i, each in enumerate(orderedFeatures):
            findSum = findSum + element[each+'_exact']
            if (i == 0):
                minimumValFind = element[each+'_exact']
                maximumValFind = element[each+'_exact']
            if (element[each+'_exact'] >= maximumValFind):
                maximumValFind = element[each+'_exact']
            if (element[each+'_exact'] <= minimumValFind):
                minimumValFind = element[each+'_exact']
        for i, each in enumerate(orderedFeatures):
            resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_scaled'] = (element[each+'_exact'] / findSum) * 70
            resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_scaledHover'] = round((element[each+'_exact'] / findSum) * 100, 2)
            if (maximumValFind == minimumValFind):
                resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_normalized'] = (1 - 0.25) * (element[each+'_exact']) + 0.25
            else:    
                resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_normalized'] = (1 - 0.25) * ((element[each+'_exact'] - minimumValFind) / (maximumValFind - minimumValFind)) + 0.25
            if (element[each].index(element[each+'_exact']) == 1):
                resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_colorScale'] = 'rgb(90, 174, 97)'
            else:  
                resultsAnalysis.loc[resultsAnalysis.index[loop], each+'_colorScale'] = 'rgb(153, 112, 171)'

    gatherStacks = []
    gatherStacksHover = []
    gatherColorsStack = []
    gatherOpacitiesStack = []
    for idx in [i for i, c in enumerate(resultsAnalysis.columns) if c.endswith("_scaled")]:
        gatherStacks.append(resultsAnalysis.iloc[:, idx].tolist())
    for idx in [i for i, c in enumerate(resultsAnalysis.columns) if c.endswith("_scaledHover")]:
        gatherStacksHover.append(resultsAnalysis.iloc[:, idx].tolist())
    for idx in [i for i, c in enumerate(resultsAnalysis.columns) if c.endswith("_colorScale")]:      
        gatherColorsStack.append(resultsAnalysis.iloc[:, idx].tolist())
    for idx in [i for i, c in enumerate(resultsAnalysis.columns) if c.endswith("_normalized")]:      
        gatherOpacitiesStack.append(resultsAnalysis.iloc[:, idx].tolist())

    collectDecisionsPerModel = collectDecisionsPerModel.to_json(double_precision=15)
    collectStatisticsPerModel = collectStatisticsPerModel.to_json(double_precision=15)
    collectInfoPerModelPandas = collectInfoPerModelPandas.to_json(double_precision=15)
    collectRoundingPandas = collectRoundingPandas.to_json(double_precision=15)
    collectLocationsAllPerSorted = collectLocationsAllPerSorted.to_json(double_precision=15)
    buildGrid = buildGrid.to_json(double_precision=15)

    metrics = metrics.to_json(double_precision=15)
    parameters = parameters.to_json(double_precision=15)

    warn = []
    for k, v in resultsAnalysis.iterrows():
        if (v['GT'] == v['prediction']):
            warn.append('rgb(255, 255, 255)')
        else:
            warn.append('rgb(255, 255, 255)')

    global decisionstatsBestRuleOver
    decisionstatsBestRuleOver = decisionstatsBest.copy(deep=True)

    removalLabels = []
    for i, el in decisionstatsBest.iterrows():
        if (i % 2 == 0):
            removalLabels.append(i)
    
    decisionstatsBestRuleOver = decisionstatsBestRuleOver.drop(decisionstatsBestRuleOver.index[removalLabels]) 
    decisionstatsBestRuleOver = decisionstatsBestRuleOver.sort_values(['purityComb','probamulalpha'], ascending=[True, False])
    decisionstatsBestRuleOver['purityComb'] *= -1

    decisionstatsBestRuleOver = decisionstatsBestRuleOver.reset_index(drop=True)

    global indexListRuleOver
    indexListRuleOver = decisionstatsBestRuleOver['gridpos'].tolist()
    purityListRuleOver = decisionstatsBestRuleOver['purityComb'].tolist()
    probamulalphaListRuleOver = decisionstatsBestRuleOver['probamulalpha'].tolist()
    print(purityListRuleOver)
    gridPosRuleOver = []
    print(indexListRuleOver)
    for element in indexListRuleOver:
        gridPosRuleOver.append(str(element+1))
    
    decisionstatsBestRuleOver = decisionstatsBestRuleOver.set_index('gridpos')

    global DataForUMAP
    DataForUMAP = pd.DataFrame(0, index=np.arange(len(X_trainRounded.index)), columns=decisionstatsBestRuleOver.index)

    for index, elem in enumerate(indexListRuleOver):
        for i, item in X_trainRounded.iterrows():
            if (item[decisionstatsBestRuleOver.loc[elem,'feature']] > decisionstatsBestRuleOver.loc[elem,'valueLimit']):
                 DataForUMAP.loc[i, elem]= 1

    # for i, item in X_trainRounded.iterrows():
    
    threshold = []    
    class0InstList = []
    class1InstList = []
    symbolUMAPBorderList = []
    for index, el in decisionstatsBestRuleOver.iterrows():
        class0Inst = []
        class1Inst = []
        symbolUMAPBorder = []    
        threshold.append(round(el['valueLimit'],keepRoundingLevel))
        for i, item in X_trainRounded.iterrows():
            if (item[el['feature']] < el['valueLimit']):
                if (y_train[i] == 1):
                    symbolUMAPBorder.append('rgb(166, 219, 160)')
                else:
                    symbolUMAPBorder.append('rgb(194, 165, 207)')        
            else:
                if (y_train[i] == 1):
                    symbolUMAPBorder.append('rgb(27, 120, 55)')
                else:
                    symbolUMAPBorder.append('rgb(118, 42, 131)')
            if (y_train[i] == 1):
                class0Inst.append(item[el['feature']])
            else:
                class1Inst.append(item[el['feature']])
        class0InstList.append(class0Inst)
        class1InstList.append(class1Inst)
        symbolUMAPBorderList.append(symbolUMAPBorder)

    if (keepRoundingLevel == 1):
        askDigits = 1 
    elif (keepRoundingLevel == 2):
        askDigits = 10 
    else:
        askDigits = 20

    num_bins = 2 * askDigits
    bin_width = 0.5 / askDigits

    binsValues = []
    for i in range(0,num_bins):
        binsValues.append(bin_width*i + bin_width)

    binsValuesFormatted = ["{:.2f}".format(value) for value in binsValues]

    valuesHist0List = []
    valuesHist1List = []
    for index in range(len(decisionstatsBestRuleOver.index)):
        hist0, edges = np.histogram(
            class0InstList[index],
            bins=num_bins,
            range=(0, bin_width*num_bins),
            density=False)
        hist1, edges = np.histogram(
            class1InstList[index],
            bins=num_bins,
            range=(0, bin_width*num_bins),
            density=False)
            
        valuesHist0List.append(hist0.tolist())
        valuesHist1List.append(hist1.tolist())


    # X_trainCopy = X_trainRounded.copy(deep=True)
    # for i, item in X_trainRounded.iterrows():
    #     for ind, feat in enumerate(orderedFeatures):
    #         idx = bisect.bisect_right(perFeatureUnique[ind],item[feat].round(r))-1
    #         if (item[feat].round(r) == 1):
    #             store = idx
    #         else:
    #             store = idx + 1
    #         X_trainCopy.loc[i,feat] = store

    neighbors = 2
    # X_embedded is your data
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(DataForUMAP)
    distances, indices = nbrs.kneighbors(DataForUMAP)
    distances = np.sort(distances, axis=0)
    distancesDec = distances[:,1]

    kneedle = KneeLocator(range(1,len(distancesDec)+1),  #x values
                        distancesDec, # y values
                        S=1.0, #parameter suggested from paper
                        curve="convex", #parameter from figure
                        direction="decreasing") #parameter from figure

    epsilon = kneedle.knee_y
    if (epsilon is None):
        epsilon = 0.5
    if (epsilon == 0.0):
        epsilon = 0.01
    if (epsilon >= 1.0):
        epsilon = 0.99
    minSamples = range(1,(len(DataForUMAP.columns)*2)+1)
    storePosition = -1
    findMax = 0
    for i in minSamples:
        db = DBSCAN(eps=epsilon, min_samples=i, n_jobs=-1).fit(DataForUMAP)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        if ((adjusted_rand_score(y_train, labels)) > findMax):
            findMax = adjusted_rand_score(y_train, labels)
            storePosition = i

    db = DBSCAN(eps=epsilon, min_samples=storePosition, n_jobs=-1).fit(DataForUMAP)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print(findMax)

    if (n_clusters_ > 25):
        n_clusters_ = 26
    elif (n_clusters_ == 1):
        n_clusters_ = 2
    else:
        pass
    
    global ModelSpaceUMAP
    ModelSpaceUMAP = FunUMAP(DataForUMAP, n_clusters_, True, [])

    # gather the results and send them back
    resultsAB.append(json.dumps(target_names)) # 0
    resultsAB.append(json.dumps(keepOriginalFeatures)) # 1
    resultsAB.append(df_cv_results_classifiers['mean_test_accuracy'].index.tolist()) # 2
    resultsAB.append(parameters) # 3 
    resultsAB.append(metrics) # 4
    resultsAB.append(json.dumps(collectDecisionsPerModel)) # 5
    resultsAB.append(json.dumps(collectStatisticsPerModel)) # 6
    resultsAB.append(json.dumps(collectInfoPerModelPandas)) # 7
    resultsAB.append(json.dumps(collectLocationsAllPerSorted)) # 8
    resultsAB.append(json.dumps(collectRoundingPandas)) # 9
    resultsAB.append(json.dumps(buildGrid)) # 10
    resultsAB.append(geometry) # 11
    resultsAB.append(perFeatureUnique) # 12
    resultsAB.append(orderedFeatures) # 13
    resultsAB.append(keepMaxValueList) # 14
    resultsAB.append(probabilitiesGen) # 15
    resultsAB.append(colorsGen) # 16
    resultsAB.append(alphaGen) # 17
    resultsAB.append(splitGen) # 18
    resultsAB.append(colorsSplitGen) # 19
    resultsAB.append(ModelSpaceUMAP) # 20
    resultsAB.append(X_trainRounded.to_json(double_precision=15)) # 21
    resultsAB.append(y_train) # 22
    resultsAB.append(X_testRounded.to_json(double_precision=15)) # 23
    resultsAB.append(y_test) # 24
    resultsAB.append(GTTestCol) # 25
    resultsAB.append(PredTestCol) # 26
    resultsAB.append(diffTest) # 27
    resultsAB.append(index_list) # 28
    resultsAB.append(rectWidths) # 29
    resultsAB.append(gatherStacks) # 30
    resultsAB.append(gatherColorsStack) # 31
    resultsAB.append(gatherOpacitiesStack) # 32
    resultsAB.append(warn) # 33
    resultsAB.append(gridPosRuleOver) # 34
    resultsAB.append(purityListRuleOver) # 35
    resultsAB.append(probamulalphaListRuleOver) # 36
    resultsAB.append(symbolUMAPBorderList) # 37
    resultsAB.append(valuesHist0List) # 38
    resultsAB.append(valuesHist1List) # 39
    resultsAB.append(threshold) # 40
    resultsAB.append(binsValuesFormatted) # 41
    resultsAB.append(TestIDs) # 42
    resultsAB.append(rectWidthsRemaining) # 43
    resultsAB.append(gatherStacksHover) # 44
    resultsAB.append(keepRoundingLevel) # 45

    SendEachClassifiersPerformanceToVisualize()

    return 'Everything Okay'

def FunUMAP (data, neighbors, flagLoc, dataTest):
    if (flagLoc):
        trans = umap.UMAP(n_neighbors=neighbors, min_dist=0.5, random_state=RANDOM_SEED, transform_seed=RANDOM_SEED).fit(data)
        Xpos = trans.embedding_[:, 0].tolist()
        Ypos = trans.embedding_[:, 1].tolist()
        return [Xpos,Ypos]
    else:
        trans = umap.UMAP(n_neighbors=neighbors, min_dist=0.5, random_state=RANDOM_SEED, transform_seed=RANDOM_SEED).fit(data)
        transTest = trans.transform(dataTest)
        Xpos = transTest[:, 0].tolist()
        Ypos = transTest[:, 1].tolist()
        return [Xpos,Ypos]



# Sending each model's results to frontend
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/PerformanceForEachModel', methods=["GET", "POST"])
def SendEachClassifiersPerformanceToVisualize():

    loopEmpty = []
    initialStates = []
    for x in range(randomSearchVar*2):
        if (x < (randomSearchVar*2/2)):
            loopEmpty.append(x+1)
        initialStates.append(1)
    response = { 
        'surrogateModelData': resultsAB,
        'X_train': X_train.to_json(double_precision=15),
        'y_train': y_train,
        'X_test': X_test.to_json(double_precision=15),
        'y_test': y_test,
    }
    return jsonify(response)

# location = './cachedir'
# memory = Memory(location, verbose=0)

# @memory.cache
# def DecisionTreeComposer(XData, X_train, y_train, X_test, clf, params, eachAlgor, AlgorithmsIDsEnd, crossValidation, randomS, RANDOM_SEED, roundValue):
#     print('insideABNow!!!')
#     # this is the grid we use to train the models
#     randSear = GridSearchCV(    
#         estimator=clf, param_distributions=params, n_iter=randomS,
#         cv=[(slice(None), slice(None))], refit='accuracy', scoring=scoring,
#         verbose=0, n_jobs=-1, random_state=RANDOM_SEED)

#     # fit and extract the probabilities
#     randSear.fit(X_train, y_train)

#     # process the results
#     cv_results = []
#     cv_results.append(randSear.cv_results_)
#     df_cv_results = pd.DataFrame.from_dict(cv_results)

#     number_of_models = []
#     # number of models stored
#     number_of_models = len(df_cv_results.iloc[0][0])

#     # initialize results per row
#     df_cv_results_per_row = []

#     modelsIDs = []
#     for i in range(number_of_models):
#         number = AlgorithmsIDsEnd+i
#         modelsIDs.append(eachAlgor+str(number))
#         df_cv_results_per_item = []
#         for column in df_cv_results.iloc[0]:
#             df_cv_results_per_item.append(column[i])
#         df_cv_results_per_row.append(df_cv_results_per_item)

#     df_cv_results_classifiers = pd.DataFrame()
#     # store the results into a pandas dataframe
#     df_cv_results_classifiers = pd.DataFrame(data = df_cv_results_per_row, columns= df_cv_results.columns)

#     # copy and filter in order to get only the metrics
#     metrics = df_cv_results_classifiers.copy()
#     metrics = metrics.filter(['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro',]) 

#     parametersPerformancePerModel = pd.DataFrame()
#     # concat parameters and performance
#     parametersPerformancePerModel = pd.DataFrame(df_cv_results_classifiers['params'])
#     parametersPerformancePerModel = parametersPerformancePerModel.to_json(double_precision=15)

#     parametersLocal = json.loads(parametersPerformancePerModel)['params'].copy()
#     Models = []
#     for index, items in enumerate(parametersLocal):
#         Models.append(str(index))

#     parametersLocalNew = [ parametersLocal[your_key] for your_key in Models ]

#     perModelProb = []
#     confuseFP = []
#     confuseFN = []
#     featureImp = []
#     collectDecisionsPerModel = pd.DataFrame()
#     collectDecisions = []
#     collectDecisionsMod = []
#     collectLocationsAll = []
#     collectStatistics = []
#     collectStatisticsMod = []
#     collectStatisticsPerModel = []
#     collectInfoPerModel = []
#     yPredictTestList = []
#     perModelPrediction = []
#     storeTrain = []
#     storePredict = []
    
#     featureNames = []
#     featureNamesDuplicated = []
    
#     for col in XData.columns:
#         featureNames.append(col)
#         featureNamesDuplicated.append(col+'_minLim')
#     for col in XData.columns:
#         featureNamesDuplicated.append(col+'_maxLim')
    
#     counterModels = 1
#     for eachModelParameters in parametersLocalNew:
#         collectDecisions = []
#         collectLocations = []
#         collectStatistics = []
#         sumRes = 0
#         clf.set_params(**eachModelParameters)
#         np.random.seed(RANDOM_SEED) # seeds
#         clf.fit(X_train, y_train) 
#         yPredictTest = clf.predict(X_test)
#         yPredictTestList.append(yPredictTest)

#         feature_importances = clf.feature_importances_
#         feature_importances[np.isnan(feature_importances)] = 0
#         featureImp.append(list(feature_importances))

#         yPredict = cross_val_predict(clf, X_train, y_train, cv=crossValidation)
#         yPredict = np.nan_to_num(yPredict)
#         perModelPrediction.append(yPredict)

#         yPredictProb = cross_val_predict(clf, X_train, y_train, cv=crossValidation, method='predict_proba')
#         yPredictProb = np.nan_to_num(yPredictProb)
#         perModelProb.append(yPredictProb.tolist())

#         storeTrain.append(y_train)
#         storePredict.append(yPredict)
#         cnf_matrix = confusion_matrix(y_train, yPredict)
#         FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
#         FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
#         FP = FP.astype(float)
#         FN = FN.astype(float)
#         confuseFP.append(list(FP))
#         confuseFN.append(list(FN))
#         for tree_idx, est in enumerate(clf.estimators_):
#             decisionPath = extractDecisionInfo(est,counterModels,tree_idx,X_train,y_train,featureNamesDuplicated,eachAlgor,feature_names=featureNames,only_leaves=True)
#             if (roundValue == -1):
#                 pass
#             else:
#                 decisionPath[0] = decisionPath[0].round(roundValue)
#             collectDecisions.append(decisionPath[0])
#             collectStatistics.append(decisionPath[1])
#             sumRes = sumRes + decisionPath[2]
#             collectLocations.append(decisionPath[3])
#         collectDecisionsMod.append(collectDecisions)
#         collectStatisticsMod.append(collectStatistics)
#         collectInfoPerModel.append(sumRes)
#         collectLocationsAll.append(collectLocations)
#         counterModels = counterModels + 1

#     collectInfoPerModelPandas = pd.DataFrame(collectInfoPerModel)

#     totalfnList = []
#     totalfpList = []
#     numberClasses = [y_train.index(x) for x in set(y_train)]
#     if (len(numberClasses) == 2):
#         for index,nList in enumerate(storeTrain):
#             fnList = []
#             fpList = []
#             for ind,el in enumerate(nList):
#                 if (el==1 and storePredict[index][ind]==0):
#                     fnList.append(ind)
#                 elif (el==0 and storePredict[index][ind]==1):
#                     fpList.append(ind)
#                 else:
#                     pass   
#             totalfpList.append(fpList)
#             totalfnList.append(fnList)
#     else:
#         for index,nList in enumerate(storeTrain):
#             fnList = []
#             class0fn = []
#             class1fn = []
#             class2fn = []
#             for ind,el in enumerate(nList):
#                 if (el==0 and storePredict[index][ind]==1):
#                     class0fn.append(ind)
#                 elif (el==0 and storePredict[index][ind]==2):
#                     class0fn.append(ind)
#                 elif (el==1 and storePredict[index][ind]==0):
#                     class1fn.append(ind)
#                 elif (el==1 and storePredict[index][ind]==2):
#                     class1fn.append(ind)
#                 elif (el==2 and storePredict[index][ind]==0):
#                     class2fn.append(ind)
#                 elif (el==2 and storePredict[index][ind]==1):
#                     class2fn.append(ind)
#                 else:
#                     pass  
#             fnList.append(class0fn)
#             fnList.append(class1fn)
#             fnList.append(class2fn)
#             totalfnList.append(fnList)
#         for index,nList in enumerate(storePredict):
#             fpList = []
#             class0fp = []
#             class1fp = []
#             class2fp = []
#             for ind,el in enumerate(nList):
#                 if (el==0 and storeTrain[index][ind]==1):
#                     class0fp.append(ind)
#                 elif (el==0 and storeTrain[index][ind]==2):
#                     class0fp.append(ind)
#                 elif (el==1 and storeTrain[index][ind]==0):
#                     class1fp.append(ind)
#                 elif (el==1 and storeTrain[index][ind]==2):
#                     class1fp.append(ind)
#                 elif (el==2 and storeTrain[index][ind]==0):
#                     class2fp.append(ind)
#                 elif (el==2 and storeTrain[index][ind]==1):
#                     class2fp.append(ind)
#                 else:
#                     pass  
#             fpList.append(class0fp)
#             fpList.append(class1fp)
#             fpList.append(class2fp)
#             totalfpList.append(fpList)
    
#     summarizeResults = []
#     summarizeResults = metrics.sum(axis=1)
#     summarizeResultsFinal = []
#     for el in summarizeResults:
#         summarizeResultsFinal.append(round(((el * 100)/3),2))

#     indices, L_sorted = zip(*sorted(enumerate(summarizeResultsFinal), key=itemgetter(1)))
#     indexList = list(indices)

#     collectDecisionsSorted = []
#     collectStatisticsSorted = []
#     collectLocationsAllSorted = []
#     for el in indexList:
#         for item in collectDecisionsMod[el]:
#             collectDecisionsSorted.append(item)
#         for item2 in collectStatisticsMod[el]:
#             collectStatisticsSorted.append(item2)
#         for item3 in collectLocationsAll[el]:
#             collectLocationsAllSorted.append(item3)

#     collectDecisionsPerModel = pd.concat(collectDecisionsSorted)
#     collectStatisticsPerModel = pd.concat(collectStatisticsSorted)
#     collectLocationsAllPerSorted = pd.DataFrame(collectLocationsAllSorted)

#     collectDecisionsPerModel = collectDecisionsPerModel.reset_index(drop=True)
#     collectStatisticsPerModel = collectStatisticsPerModel.reset_index(drop=True) 
#     collectLocationsAllPerSorted = collectLocationsAllPerSorted.reset_index(drop=True) 
#     collectDecisionsPerModel = collectDecisionsPerModel.to_json(double_precision=15)
#     collectStatisticsPerModel = collectStatisticsPerModel.to_json(double_precision=15)
#     collectInfoPerModelPandas = collectInfoPerModelPandas.to_json(double_precision=15)
#     collectLocationsAllPerSorted = collectLocationsAllPerSorted.to_json(double_precision=15)

#     perModelPredPandas = pd.DataFrame(perModelPrediction)
#     perModelPredPandas = perModelPredPandas.to_json(double_precision=15)

#     yPredictTestListPandas = pd.DataFrame(yPredictTestList)
#     yPredictTestListPandas = yPredictTestListPandas.to_json(double_precision=15)

#     perModelProbPandas = pd.DataFrame(perModelProb)
#     perModelProbPandas = perModelProbPandas.to_json(double_precision=15)

#     metrics = metrics.to_json(double_precision=15)
#     # gather the results and send them back
#     resultsAB.append(modelsIDs) # 0 17
#     resultsAB.append(parametersPerformancePerModel) # 1 18
#     resultsAB.append(metrics) # 2 19
#     resultsAB.append(json.dumps(confuseFP)) # 3 20
#     resultsAB.append(json.dumps(confuseFN)) # 4 21
#     resultsAB.append(json.dumps(featureImp)) # 5 22
#     resultsAB.append(json.dumps(collectDecisionsPerModel)) # 6 23
#     resultsAB.append(perModelProbPandas) # 7 24
#     resultsAB.append(json.dumps(perModelPredPandas)) # 8 25
#     resultsAB.append(json.dumps(target_names)) # 9 26
#     resultsAB.append(json.dumps(collectStatisticsPerModel)) # 10 27
#     resultsAB.append(json.dumps(collectInfoPerModelPandas)) # 11 28
#     resultsAB.append(json.dumps(keepOriginalFeatures)) # 12 29
#     resultsAB.append(json.dumps(yPredictTestListPandas)) # 13 30
#     resultsAB.append(json.dumps(collectLocationsAllPerSorted)) # 14 31
#     resultsAB.append(json.dumps(totalfpList)) # 15 32
#     resultsAB.append(json.dumps(totalfnList)) # 16 33

#     return resultsAB


def extractDecisionInfo(decision_tree,counterModels,tree_index,X_train,y_predict,feature_names_duplicated,listOfRules,feature_names=None,only_leaves=True):
    '''return dataframe with node info
    '''
    #decision_tree.fit(X_train, y_predict)
    # extract info from decision_tree

    rulesNumber = get_rules_proba(decision_tree, XData.columns, target_names)
    listOfRules.append(rulesNumber)
    n_nodes = decision_tree.tree_.node_count
    children_left = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    feature = decision_tree.tree_.feature
    threshold = decision_tree.tree_.threshold
    impurity = decision_tree.tree_.impurity
    value = decision_tree.tree_.value
    n_node_samples = decision_tree.tree_.n_node_samples
    whereTheyBelong = decision_tree.apply(X_train)

    # cast X_train as dataframe
    df = pd.DataFrame(X_train)
    if feature_names is not None:
        df.columns = feature_names
    
    # indexes with unique nodes
    idx_list = df.assign(
        leaf_id = lambda df: decision_tree.apply(df)
    )[['leaf_id']].drop_duplicates().index

    # test data for unique nodes
    X_test = df.loc[idx_list,].to_numpy()
    # decision path only for leaves
    dp = decision_tree.decision_path(X_test)
    # final leaves for each data
    leave_id = decision_tree.apply(X_test)
    # values for each data
    leave_predict = decision_tree.predict(X_test)
    # dictionary for leave_id and leave_predict
    dict_leaves = {k:v for k,v in zip(leave_id,leave_predict)}
    
    # create decision path information for all nodes
    dp_idxlist = [[ini, fin] for ini,fin in zip(dp.indptr[:-1],dp.indptr[1:])]
    dict_decisionpath = {}
    for idxs in dp_idxlist:
        dpindices = dp.indices[idxs[0]:idxs[1]]
        for i,node in enumerate(dpindices):
            if node not in dict_decisionpath.keys():
                dict_decisionpath[node] = dpindices[:i+1]
    
    # initialize number of columns and output dataframe
    n_cols = df.shape[-1]
    df_thr_all = pd.DataFrame()

    # predict for samples
    for node, node_index in dict_decisionpath.items():
        l_thresh_max = np.ones(n_cols) * np.nan
        l_thresh_min = np.ones(n_cols) * np.nan
        
        # decision path info for each node
        for i,node_id in enumerate(node_index):
            if node == node_id:
                continue

            if children_left[node_id] == node_index[i+1]: #(X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                l_thresh_max[feature[node_id]] = threshold[node_id]
            else:
                l_thresh_min[feature[node_id]] = threshold[node_id]
        # append info to df_thr_all
        df_thr_all = df_thr_all.append(
            [[thr_min for thr_max,thr_min in zip(l_thresh_max,l_thresh_min)]
             + [thr_max for thr_max,thr_min in zip(l_thresh_max,l_thresh_min)]
             + [
                 node,
                 counterModels,
                 tree_index,
                 np.nan if node not in dict_leaves.keys() else dict_leaves[node],
                 #value[node],
                 impurity[node],
                 n_node_samples[node],
               ]
            ]
        )
    # rename columns and set index
    if feature_names is not None:
        df_thr_all.columns = feature_names_duplicated + ['node','counterModels','tree_index','predicted_value','impurity','samples']
    else:
        df_thr_all.columns = ['X_{}'.format(i) for i in range(n_cols)] + ['node','counterModels','tree_index','predicted_value','impurity','samples']
    #df_thr_all = df_thr_all.set_index('decision')
    #df_thr_all = df_thr_all.reset_index(drop=True)
    if only_leaves:
        df_thr_all = df_thr_all[~df_thr_all['predicted_value'].isnull()]
        df_thr_all['impurity'].loc[df_thr_all['impurity'] < 0] = 0
        df_thr_all['impurity'].loc[:] = round((df_thr_all['impurity'].loc[:]) * 100, 2)
        # df_thr_all['impurity'].loc[df_thr_all['impurity'] >= 0.5] = 0.8

    # del df_thr_all['decision']
    # del df_thr_all['predicted_value']

    #df_thr_all.reset_index()

    df_thr_all = df_thr_all.replace(np.nan,2) # nan mapped as value 2

    #df_thr_all = df_thr_all.sort_index()
    
    copy_df_thr_all = df_thr_all.copy()

    del df_thr_all['node']
    del df_thr_all['counterModels']
    del df_thr_all['tree_index']
    del df_thr_all['predicted_value']
    del df_thr_all['impurity']
    del df_thr_all['samples']

    copy_df_thr_all = copy_df_thr_all[['node','counterModels','tree_index','predicted_value', 'impurity', 'samples']]
    return [df_thr_all,copy_df_thr_all,len(df_thr_all),whereTheyBelong,listOfRules]

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/updateUMAP', methods=["GET", "POST"])
def updateUMAPFun():
    RetrievedUpdatedRule = request.get_data().decode('utf8').replace("'", '"')
    RetrievedUpdatedRule = json.loads(RetrievedUpdatedRule)

    global checkingRule
    global thresholdRule

    checkingRule = int(RetrievedUpdatedRule['checkingRule'])
    localRule = int(RetrievedUpdatedRule['localRule'])
    thresholdRule = float(RetrievedUpdatedRule['thresholdRule'])

    # print(checkingRule)
    # print(localRule)
    # print(thresholdRule)

    if (thresholdRule <= 0):
        thresholdRule = 0
    if (thresholdRule >= 1):
        thresholdRule = 1

    DataForUMAPTest = pd.DataFrame(0, index=np.arange(len(X_trainRounded.index)), columns=decisionstatsBestRuleOver.index)
    decisionstatsBestRuleOverModified = copy.deepcopy(decisionstatsBestRuleOver)

    decisionstatsBestRuleOverModified.loc[localRule,'valueLimit'] = round(thresholdRule,keepRoundingLevel)

    for index, elem in enumerate(indexListRuleOver):
        for i, item in X_trainRounded.iterrows():
            if (item[decisionstatsBestRuleOverModified.loc[elem,'feature']] > decisionstatsBestRuleOverModified.loc[elem,'valueLimit']):
                 DataForUMAPTest.loc[i, elem]= 1
    # print(DataForUMAP)
    symbolUMAPBorder = []
    class0Inst = []
    class1Inst = []
    for i, item in X_trainRounded.iterrows():
        if (item[decisionstatsBestRuleOverModified.loc[localRule,'feature']] < decisionstatsBestRuleOverModified.loc[localRule,'valueLimit']):
            if (y_train[i] == 1):
                symbolUMAPBorder.append('rgb(166, 219, 160)')
            else:
                symbolUMAPBorder.append('rgb(194, 165, 207)')        
        else:
            if (y_train[i] == 1):
                symbolUMAPBorder.append('rgb(27, 120, 55)')
            else:
                symbolUMAPBorder.append('rgb(118, 42, 131)')
        if (y_train[i] == 1):
            class0Inst.append(item[decisionstatsBestRuleOverModified.loc[localRule,'feature']])
        else:
            class1Inst.append(item[decisionstatsBestRuleOverModified.loc[localRule,'feature']])
            
    global DataForUMAP

    neighbors = 2
    # X_embedded is your data
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(DataForUMAP)
    distances, indices = nbrs.kneighbors(DataForUMAP)
    distances = np.sort(distances, axis=0)
    distancesDec = distances[:,1]

    kneedle = KneeLocator(range(1,len(distancesDec)+1),  #x values
                        distancesDec, # y values
                        S=1.0, #parameter suggested from paper
                        curve="convex", #parameter from figure
                        direction="decreasing") #parameter from figure

    epsilon = kneedle.knee_y
    if (epsilon is None):
        epsilon = 0.5
    if (epsilon == 0.0):
        epsilon = 0.01
    if (epsilon >= 1.0):
        epsilon = 0.99
    minSamples = range(1,(len(DataForUMAP.columns)*2)+1)
    storePosition = -1
    findMax = 0
    for i in minSamples:
        db = DBSCAN(eps=epsilon, min_samples=i, n_jobs=-1).fit(DataForUMAP)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        if ((adjusted_rand_score(y_train, labels)) > findMax):
            findMax = adjusted_rand_score(y_train, labels)
            storePosition = i

    db = DBSCAN(eps=epsilon, min_samples=storePosition, n_jobs=-1).fit(DataForUMAP)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print(findMax)

    if (n_clusters_ > 25):
        n_clusters_ = 26
    elif (n_clusters_ == 1):
        n_clusters_ = 2
    else:
        pass

    ModelSpaceUMAPUpdated = FunUMAP(DataForUMAP, n_clusters_, False, DataForUMAPTest)
    global ModelSpaceUMAP
    #mtx1, mtx2, disparity = procrustes(ModelSpaceUMAP, ModelSpaceUMAPUpdated)
    # print(disparity)
    #ModelSpaceUMAPUpdated = mtx2.tolist()
    #ModelSpaceUMAPLocal = mtx1.tolist()

    gatherData = []
    gatherData.append(ModelSpaceUMAPUpdated)
    gatherData.append(y_train)
    gatherData.append(symbolUMAPBorder)
    gatherData.append(ModelSpaceUMAP)

    response = { 
        'updateUMAP': gatherData,
    }

    return jsonify(response)

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/callRecomputeModel', methods=["GET", "POST"])
def updateModelFun():
    
    executeSearch(False,True)

    return 'okay'

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/callRecomputeRuleModel', methods=["GET", "POST"])
def updateRuleModelFun():
    
    RetrievedUpdatedRuleModel = request.get_data().decode('utf8').replace("'", '"')
    RetrievedUpdatedRuleModel = json.loads(RetrievedUpdatedRuleModel)

    global bestModel

    bestModel = int(RetrievedUpdatedRuleModel['recomputeRuleModelVar'])

    executeSearch(True,False)

    return 'okay'

def procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.
    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.
    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets
    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.
    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.
    - The disparity scales as the number of points per input matrix.
    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
    Examples
    --------
    >>> from scipy.spatial import procrustes
    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:
    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0.0
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    # mtx1 -= np.mean(mtx1, 0)
    # mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    # if norm1 == 0 or norm2 == 0:
    #     raise ValueError("Input matrices must contain >1 unique points")

    # # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity