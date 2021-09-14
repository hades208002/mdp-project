# Stacking Model

# description -> simple implementation
'''
- the central model must have be able to send requests to the local models -> array of models
- different aggregation methods (to take into account the probability distribution, provide better visualizations)
'''

# procedure
'''
1) send the data to the single models -> in this simple application -> invoque the functions through the array of models (what if a new model wants to be added? -> right now, we just want to check if the stacking procedure is working fine!!)
2) combine the responce
3) predict (aggregating somehow the prediction of the other models)

'''
### ADD CROSS-VALIDATION TO CHOOSE THE BEST AGGREGATIN METHOD

# Import all the useful libraries
from LocalModel import LocalModel

import numpy as np
import pandas as pd
import fancyimpute
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier  # PROBABILITY
from sklearn.tree import DecisionTreeClassifier  # PROBABILITY
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier  # PROBABILITY
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier  # PROBABILITY
from sklearn.linear_model import LogisticRegression  # PROBABILITY
from sklearn.naive_bayes import GaussianNB  # PROBABILITY
from sklearn.ensemble import ExtraTreesClassifier  # PROBABILITY
from sklearn.neighbors import KNeighborsClassifier  # PROBABILITY
from sklearn.ensemble import BaggingClassifier  # PROBABILITY

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks


class CentralModel(LocalModel):

    # initialize
    def __init__(self, data, target_name, model_name="dt4", random_state=12345678, imputation_strategy='mice',
                 balance_strategy='SMOTE', local_models=[]):
        LocalModel.__init__(self, data=data, target_name=target_name, model_name=model_name, random_state=random_state,
                            imputation_strategy=imputation_strategy, balance_strategy=balance_strategy)
        self.local_models = local_models
        self.new_dataframe = None  # the original dataframe (used for trainig) plus the predictions of the local models

    # add a new local model to the set of local models
    def addLocalModel(self, localModel):
        if type(localModel) is list:
            self.local_models = self.local_models + localModel
        else:
            self.local_models.append(localModel)

    def prepareDataset(self, train_x):
        local_predictions = {}
        for i in self.local_models:
            train_x_copy = train_x.copy()
            name = i.selected_model_name
            print("NAME ", name)
            prediction = i.predict(train_x_copy)
            if 'prediction' in prediction.columns:
                local_predictions['prediction ' + name] = prediction['prediction']
            if 'predict_proba_zero' in prediction.columns:
                local_predictions["predict_proba_zero " + name] = prediction['predict_proba_zero']
            if 'predict_proba_uno' in prediction.columns:
                local_predictions["predict_proba_uno " + name] = prediction['predict_proba_uno']
            if 'predict_log_proba_zero' in prediction.columns:
                local_predictions["predict_log_proba_zero " + name] = prediction['predict_log_proba_zero']
            if 'predict_log_proba_uno' in prediction.columns:
                local_predictions["predict_log_proba_uno " + name] = prediction['predict_log_proba_uno']
        ## add other features (or remove them) as necessary
        ## also need to update the local model.predict function , if somehing is modified here
        ## ...
        # 2) combine the prediction in a dataframe
        local_predictions = pd.DataFrame(local_predictions)
        train_x = train_x.reset_index()  ## reset the index -> begin from zero -> in this case we can cobine the two datasets (train_x and local_predictions)
        train_x = train_x.drop('index', axis=1)
        train_dataframe_x = pd.concat([train_x, local_predictions], axis=1)
        train_dataframe_x = train_dataframe_x.replace(-float('Inf'), -9999999)  # cannot deal with -inf
        return train_dataframe_x

    # train the central model
    def train(self):
        print("BEGIN CENTRAL TRAINING")
        # 0) clean and balance the training dataset -> done in the init phase
        train_x = self.data[self.features]  ## data to send to the local models
        train_y = self.data[self.target_name]  ## target
        # 1) send the data to the local models and wait for their predictions
        # if there is at least one local model
        if self.local_models != []:
            train_dataframe_x = self.prepareDataset(train_x)
            # 3) train the central model on the new dataframe

            train_y = train_y.reset_index().drop('index', axis=1)
            self.selected_model.fit(train_dataframe_x, train_y)
            # self.features = train_dataframe_x.columns ## update the number of features to mech the new dataset
            self.new_dataframe = pd.concat([train_dataframe_x, train_y], axis=1)
            print("END CENTRAL TRAINING")
            return
        else:
            super(CentralModel, self).train()

    def predict(self, test):
        print("begin prediction")
        original_x = test
        train = test.copy()
        x_test = self.cleanDataTest(test_x=train)
        x_test = self.recoverMissing(data=x_test)
        result = x_test.copy()
        if self.local_models != []:
            test_dataframe_x = self.prepareDataset(x_test)
            prediction = self.selected_model.predict(test_dataframe_x)
            result['prediction'] = prediction
            return pd.DataFrame(result)
        else:
            prediction = self.selected_model.predict(x_test)
            result['prediction'] = prediction
            return pd.DataFrame(result)
        print("end prediction")