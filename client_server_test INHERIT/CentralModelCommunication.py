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

# communication
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread, RLock
import json


class CentralModelCommunication(LocalModel):
    global clientRequesting  # indicate the client who is asking to perform the prediction

    global countResponses
    countResponses = 0
    lock = RLock()

    # initialize
    def __init__(self, data="none", target_name="AFclass", model_name="none", random_state=12345678,
                 imputation_strategy='mice', balance_strategy='SMOTE', local_models=[]):
        LocalModel.__init__(self, data=data, target_name=target_name, model_name=model_name, random_state=random_state,
                            imputation_strategy=imputation_strategy, balance_strategy=balance_strategy)
        self.local_models = local_models
        self.new_dataframe = {}  # the original dataframe (used for trainig) plus the predictions of the local models
        # CENTRAL TRAINING DATA
        self.trainingData = pd.DataFrame()

        self.trainDataPredicted = pd.DataFrame()  # only the X part (no target) # data (predicted) returned from the clients -> used to train the central model
        self.trainDataPredictedChunk = {}  # only the X part (no target) # return of a chunk
        self.chunkNumber = 0  # used to send multiple chunks for prediction
        self.numberOfChunks = 0
        self.splits = []  # list of prediciton splits

        # communication
        self.stackingClients = {}  # clients used to train the central model
        self.clients = {}  ## model that can predict -> gui
        self.appClients = {}  ## no prediction capabilities
        self.clientsAccuracy = {}

        self.addresses = {}
        self.counter = 0  # count the local hospital
        self.HOST = ''
        self.PORT = 33000
        self.BUFSIZ = 102400000
        self.ADDR = (self.HOST, self.PORT)
        self.SERVER = socket(AF_INET, SOCK_STREAM)
        self.SERVER.bind(self.ADDR)
        self.SERVER.listen(10)  ## queue length
        print("Waiting for connection...")
        self.ACCEPT_THREAD = Thread(target=self.accept_incoming_connections)
        self.ACCEPT_THREAD.start()
        self.ACCEPT_THREAD.join()  # TRY TO REVOVE this line and see what happen!!

    # self.SERVER.close()
    # self.lock = Lock()


    """Sets up handling for incoming clients."""

    def accept_incoming_connections(self):
        while True:
            client, client_address = self.SERVER.accept()
            print("%s:%s has connected." % client_address)
            client.sendall(bytes("Greetings from the cave! Now type your name and press enter!", "utf8"))
            self.addresses[client] = client_address
            Thread(target=self.handle_client, args=(client,)).start()

    # establish connection and get responses
    '''
    def recv_basic(self, the_socket):
        total_data=[]
        while True:
            data = the_socket.recv(8192)
            if not data: 
                return ''.join(total_data)
            total_data.append(data)
            print ("total_data")
            print (total_data)
    '''

    def handle_client(self, client):  # Takes client socket as argument.
        """Handles a single client connection."""
        data = client.recv(self.BUFSIZ).decode("utf8")
        print(data)
        # welcome = 'Welcome %s! If you ever want to quit, type {quit} to exit.' % name
        client.sendall(bytes("connection OK", "utf8"))  # send(bytes(welcome, "utf8"))
        # self.lock.acquire()

        modelType = data[:3]
        modelAccuracy = data[3:]

        if modelType == "gui":
            self.clients[client] = "Local_hospital_" + str(self.counter)
            self.counter = self.counter + 1
            if len(modelAccuracy) > 0:
                self.clientsAccuracy[self.clients[client]] = float(modelAccuracy)

        if modelType == "app":
            self.appClients[client] = "app_client_" + str(len(self.appClients))

        # self.lock.release()
        print("CONNECTION ON")
        print("number of gui clients -> ", self.counter)
        print("number of stacking clients ->", self.stackingClients)
        print("number of app clients -> ", len(self.appClients))
        '''
        msg = 'receiving begin'
        text = ''
        chunk = ''
        while True:
            try:
                chunk += client.recv()
                if not chunk:
                    print ("message arrived entirely chunk")
                    # Unreliable
                    msg = text.decode("utf8")
                    try :
                        data = json.loads(msg)
                        if data[0] == "train":
                            data_received = pd.read_json(data[1])
                            print (" DATA RECEIVED _> : ", data_received)
                            #self.broadcast(data_received)
                            # perform training
                        elif data[0] == "predict":
                            data_received = pd.read_json(data[1])
                            print ("EOEOE 1")
                            #print (" DATA RECEIVED _> Predict: ", data_received)
                            #self.lock.acquire()
                            #print ("self.countResponses : " +  self.countResponses + " ---  self.counter : " + self.counter) 
                            print ("EOEOE 2")
                            print ("counter -> : ", self.counter)

                            global countResponses
                            print ("countResponses -> ",countResponses)
                            
                            
                            #self.broadcast(data_received)
                            if 'prediction' in data_received.columns:
                                self.new_dataframe['prediction ' +self.clients[client]]= data_received['prediction']
                            print ("EOEOE 3")
                            countResponses += 1
                            if countResponses == self.counter :
                                self.predict_on_data()
                            print ("EOEOE 4")
                            print ("countResponses -> ",countResponses)
                            #print ("self.countResponses : " +  self.countResponses + " ---  self.counter : " + self.counter) 
                                # perform prediction
                        elif data[0] == "requested_prediction":
                            self.new_dataframe = {}
                            global clientRequesting
                            clientRequesting = client
                            data_received = pd.read_json(data[1])
                            self.prepareDataset_prediction(data_received)
                    except:
                        print ("JSON ERROR")
                        #self.broadcast(msg) 
                    text = '' # reset to zero the message
                else:
                    text += chunk
            except Exception:
                if text:
                    #print ("ERROR receiving")
                    pass
                else:
                    #print ("ERROR receiving")
                    pass

        print ("MESSAGE : ", msg)  
        '''

        while True:
            try:
                msg = client.recv(self.BUFSIZ).decode("utf8")

                ## TEST
                '''
                msg = 'Something wrong in receiving'
                try:
                    text = ''
                    chunk = ''
                    while True:
                        chunk += client.recv()
                        if not chunk:
                            print ("no chunk")
                            # Unreliable
                            break
                        else:
                            text += chunk
                    msg = text.decode("utf8")
                except Exception:
                    if text:
                        pass
                    else:
                        print ("ERROR receiving")
                '''
                ## END TEST

                print("MESSAGE : ", msg)
                try:
                    global countResponses
                    data = json.loads(msg)
                    if data[0] == "train":
                        data_received = pd.read_json(data[1])
                        print(" DATA RECEIVED _> : ", data_received)
                        self.addDataFrameToTrainSet(data_received)
                        self.sendData(pd.DataFrame(), "training received", client)
                    # self.broadcast(data_received)
                    # perform training

                    elif data[0] == "features":
                        print("features received")
                        data_received = pd.read_json(data[1])
                        print(data_received)
                        self.updateFeatures(data_received)
                        self.sendData(self.important_features, "important features", client)

                    elif data[0] == "trainset predicted":
                        print("trainset predicted")
                        data_received = pd.read_json(data[1])
                        print("countResponses -> ", countResponses)
                        if 'prediction' in data_received.columns:
                            print("qwer 1")
                            self.trainDataPredictedChunk['prediction ' + self.clients[client]] = data_received[
                                'prediction']
                            print("qwer 2")
                        print("qwer 3")
                        countResponses += 1
                        print("qwer 4")
                        if countResponses == len(self.stackingClients):
                            print("qwer 5")
                            countResponses = 0
                            print("qwer 6")
                            self.train_next_chunk()
                            print("qwer 7")

                    elif data[0] == "predict":
                        data_received = pd.read_json(data[1])
                        print("EOEOE 1")
                        # print (" DATA RECEIVED _> Predict: ", data_received)
                        # self.lock.acquire()
                        # print ("self.countResponses : " +  self.countResponses + " ---  self.counter : " + self.counter)
                        print("EOEOE 2")
                        print("counter -> : ", self.counter)
                        print("countResponses -> ", countResponses)

                        # self.broadcast(data_received)

                        if 'prediction' in data_received.columns:
                            self.new_dataframe['prediction ' + self.clients[client]] = data_received['prediction']
                        print("EOEOE 3")
                        countResponses += 1
                        clientSum = len(self.clients)  # used immediately to check when to call the predict_on_data
                        print(" stackingClients len -> : ", len(self.stackingClients))
                        if len(self.stackingClients) > 0:
                            clientSum = len(self.stackingClients)
                        if countResponses == clientSum:
                            self.predict_on_data()
                        print("EOEOE 4")
                        print("countResponses -> ", countResponses)
                    # print ("self.countResponses : " +  self.countResponses + " ---  self.counter : " + self.counter)
                    # perform prediction
                    elif data[0] == "requested_prediction":
                        self.new_dataframe = {}
                        global clientRequesting
                        clientRequesting = client
                        data_received = pd.read_json(data[1])
                        self.prepareDataset_prediction(data_received)

                    elif data[0] == "trigger test":
                        self.train()
                except:
                    print("JSON ERROR")
                # self.broadcast(msg)
                # self.broadcast(msg)
            except OSError:  # Possibly client has left the chat.
                break

    '''
    def broadcast(self, msg): 
        """Broadcasts a message to all the clients."""

        for sock in self.clients:
            sock.sendall(bytes(msg, "utf8"))  #(bytes(prefix, "utf8")+msg)
            #sock.send(msg)
    '''

    def broadcast(self, data, label):
        print("BROEADAST 1")
        to_send = data.to_json()
        print("BROEADAST 2")
        message = json.dumps((label, to_send))
        print("BROEADAST 3")
        clients = self.clients
        print("BROEADAST 4")
        print("stacking clients len -> ", len(self.stackingClients))
        if len(self.stackingClients) > 0:
            clients = self.stackingClients
        for sock in clients:
            sock.sendall(bytes(message, "utf8"))
        print("sent to all -> BROEADAST 5")

    def sendData(self, data, label, client):
        to_send = data.to_json()
        message = json.dumps((label, to_send))
        client.sendall(bytes(message, "utf8"))
        print("message sent")

    # add a new local model to the set of local models
    def addLocalModel(self, localModel):
        if type(localModel) is list:
            self.local_models = self.local_models + localModel
        else:
            self.local_models.append(localModel)

    def prepareDataset_prediction(self, train_x):
        # send data to all the models
        print("PREDICTION REQUEST -> PREPARING DATA")
        '''
        to_send = train_x.to_json()
        message = json.dumps(("predict" , to_send))
        self.broadcast(message)
        '''
        self.broadcast(train_x, "predict")
        print("PREDICTION REQUEST -> PREPARING DATA")

    def prepareDataset(self, train_x):

        # send data to all the models
        '''
        to_send = train_x.to_json()
        message = json.dumps(("predict" , to_send))
        self.broadcast(message)
        '''
        self.broadcast(train_x, "predict")

        '''
        local_predictions = {}
        for i in self.local_models :
            train_x_copy = train_x.copy()
            name = i.selected_model_name
            print ("NAME ", name )
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
        
        train_x = train_x.reset_index() ## reset the index -> begin from zero -> in this case we can cobine the two datasets (train_x and local_predictions)
        train_x = train_x.drop('index', axis = 1)
        train_dataframe_x =  pd.concat([train_x, local_predictions], axis=1)
        train_dataframe_x = train_dataframe_x.replace(-float('Inf'), -9999999) # cannot deal with -inf
        return train_dataframe_x
        
        return local_predictions
        '''

    def addDataFrameToTrainSet(self, data):
        # data must be a dataframe -> with the "same" features name
        print("prepare to add DATA")
        try:
            # self.trainingData["h"] = data
            self.trainingData = pd.concat([self.trainingData, data])
            self.trainingData.sort_index(inplace=True)
        except:
            print("CANNOT ADD DATA")
        print("heieie")
        print("current CENTRAL TRAIN DATASET -> dimension : ", self.trainingData.shape)

    # train the central model

    def trainModel(self):
        print("begin model training")
        self.selected_model.fit(self.trainDataPredicted, self.target)
        print("end training")

    def train_next_chunk(self):
        print("send data to be trained")
        # update trainset
        print("wtf 1")
        self.trainDataPredicted = pd.concat([self.trainDataPredicted, pd.DataFrame(
            self.trainDataPredictedChunk)])  # data (predicted) returned from the clients -> used to train the central model
        print("wtf 2")
        self.trainDataPredictedChunk = {}  # reset to get the new data
        print("wtf 3")
        if self.chunkNumber < self.numberOfChunks:
            print("wtf 4")
            # train next chunk
            self.broadcast(self.splits[self.chunkNumber], "train")
            print("wtf 5")
            self.chunkNumber += 1
            print("wtf 6")
        else:
            print("wtf 7")
            self.splits = []
            print("wtf 8")
            self.numberOfChunks = 0
            print("wtf 9")
            self.chunkNumber = 0
            print("wtf 10")
        print("wtf 11")
        self.trainModel()

    def train(self):
        # reset
        print("begin training function 1")
        global countResponses
        countResponses = 0
        self.trainingData = self.trainingData.reset_index().drop('index', axis=1)
        self.trainDataPredicted = pd.DataFrame()  # data (predicted) returned from the clients -> used to train the central model
        self.trainDataPredictedChunk = {}  # return of a chunk
        self.chunkNumber = 0  # used to send multiple chunks for prediction
        self.numberOfChunks = 0
        self.splits = []  # list of prediciton splits
        # split the dataset
        self.stackingClients = self.clients.copy()
        print("self.stackingClients -> len : ", len(self.stackingClients))
        print("stop 1")
        train_x = self.trainingData[self.trainingData.columns[self.trainingData.columns != self.target_name]]
        print("stop 2")
        self.target = self.trainingData[self.target_name]
        print("stop 3")
        # split train_x into chunks to be sent for prediction
        self.splits = self.splitDataframe(train_x)
        print("stop 4")
        self.numberOfChunks = len(self.splits)
        print("begin training function 2")
        print("train shape -> : ", self.trainingData.shape)
        print(self.splits)
        print("self.chunkNumber -> ", self.chunkNumber)
        # send the first chuck to be predicted by the client models
        self.broadcast(self.splits[self.chunkNumber], "train")
        self.chunkNumber += 1
        # UPDATE CLIENTS TRAINED LIST -> to corret prediction phase
        print("UPDATEe CLIENTs LIST")

    '''
    def train(self):
        print ("BEGIN CENTRAL TRAINING")
        # 0) clean and balance the training dataset -> done in the init phase
        train_x = self.data[self.features] ## data to send to the local models
        ## target
        train_y = self.trainingData[self.target_name]

        # 1) send the data to the local models and wait for their predictions 
        # if there is at least one local model
        if self.local_models != []:
            train_dataframe_x = self.prepareDataset(train_x)
            # 3) train the central model on the new dataframe
            
            train_y = train_y.reset_index().drop('index', axis = 1)

            self.cv_x = train_dataframe_x
            self.cv_y = train_y
            # crossValidate
            self.chooseModel_with_crossValidation()

            self.selected_model.fit(train_dataframe_x,train_y)
            #self.features = train_dataframe_x.columns ## update the number of features to mech the new dataset 
            self.new_dataframe = pd.concat([train_dataframe_x, train_y], axis=1)
            print ("END CENTRAL TRAINING")
            return
        else :
            super(CentralModel, self).train()
    '''

    # predict once all the responces from the local models have been collected
    def predict_on_data(self):
        print("PREDICTION predict_on_data")
        print("counter : ", self.counter)
        '''
        self.lock.acquire()
        self.countResponses = self.countResponses + 1
        print ("self.countResponses : " +  self.countResponses + " ---  self.counter : " + self.counter) 
        self.lock.release()
        '''
        self.new_dataframe = pd.DataFrame(self.new_dataframe)
        print(self.new_dataframe)
        global countResponses
        countResponses = 0
        self.predict(self.new_dataframe)

    def computeWeightedVoting(self, row):
        try:
            print("local hospital -> accuracy AAA ", self.clientsAccuracy)
            model_temp = "Local_hospital_"
            result = 0
            print("local hospital -> accuracy BBB", self.clientsAccuracy)
            accuracySum = sum(self.clientsAccuracy.values)
            for i in [0, self.counter]:
                m = model_temp + str(i)
                result = result + row["prediction " + m] * self.clientsAccuracy[m]
            print("local hospital -> accuracy CCC", self.clientsAccuracy)
            return (round(result / accuracySum, 0))
        except:
            print("local hospital -> accuracy DDD", self.clientsAccuracy)
            for i in [0, self.counter]:
                m = model_temp + str(i)
                result = result + row["prediction " + m]
            print("local hospital -> accuracy EEE", self.clientsAccuracy)
            return (round(result / self.counter, 0))

    def predict(self, test):
        t = test
        if len(self.stackingClients) <= 1:
            print("majority voting")
            # no training data -> no central model -> perform majority voting
            try:
                # t['finalPrediction'] = t[t.columns].mode(axis=1)
                t['finalPrediction'] = t.apply(self.computeWeightedVoting, axis=1)
                print("pred -> majority voting SUCCEED")
            except:
                try:
                    t['finalPrediction'] = t[t.columns].mode(axis=1)
                    print("pred -> majority voting SUCCEED -> MODE ")
                except:
                    print("pred -> majority voting FAILED")
                    t['finalPrediction'] = t[t.columns].min(axis=1)  # if everything fails -> class 0 ;)
                    print("pred -> min SUCCEED")

        else:
            # sofisticated aggregation
            print("sofisticated aggregation")
            print("self.stackingClients -> ", len(self.stackingClients))
            print("traing data columns -> ", self.trainDataPredicted.shape[1])
            try:
                print("sofisticated aggregation begin")
                prediction = self.selected_model.predict(t)
                t['finalPrediction'] = prediction
                print("FINAL PREDICTUION as")
                print(t['finalPrediction'])
                t['finalPrediction'] = t['finalPrediction'].map({'persistierend (>7 Tage, EKV)': 1, 'paroxysmal': 0})
                print("sofisticated aggregation SUCCEED")
            except:
                print("sofisticated aggregation FAILED")
                print("back to majority voting")
                try:
                    t['finalPrediction'] = t[t.columns].mode(axis=1)
                    print("pred -> majority voting SUCCEED")
                except:
                    print("pred -> majority voting FAILED")
                    t['finalPrediction'] = t[t.columns].min(axis=1)  # if everything fails -> class 0 ;)
                    print("pred -> min SUCCEED")
        print(t)
        print("t printed -> send results")
        to_send = t.to_json()
        message = json.dumps(("result", to_send))
        global clientRequesting
        clientRequesting.sendall(bytes(message, "utf8"))

    # return t["finalPrediction to who has asked"]



    '''
    def predict (self, test):
        print ("begin prediction")
        original_x = test
        train = test.copy()
        x_test = self.cleanDataTest(test_x = train)
        x_test = self.recoverMissing(data = x_test)
        result = x_test.copy()
        if self.local_models != []:
            test_dataframe_x = self.prepareDataset(x_test)
            prediction = self.selected_model.predict(test_dataframe_x)
            result['prediction'] = prediction
            return pd.DataFrame(result)
        else :
            prediction = self.selected_model.predict(x_test)
            result['prediction'] = prediction
            return pd.DataFrame(result)
        print ("end prediction")
    '''
