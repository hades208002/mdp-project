from LocalModel import LocalModel


# Import all the useful libraries
import numpy as np
import pandas as pd
import fancyimpute
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold

# communication
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import json


from sklearn.ensemble import AdaBoostClassifier # PROBABILITY
from sklearn.tree import DecisionTreeClassifier # PROBABILITY
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier # PROBABILITY
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier # PROBABILITY
from sklearn.linear_model import LogisticRegression # PROBABILITY
from sklearn.naive_bayes import GaussianNB # PROBABILITY
from sklearn.ensemble import ExtraTreesClassifier # PROBABILITY
from sklearn.neighbors import KNeighborsClassifier # PROBABILITY
from sklearn.ensemble import BaggingClassifier # PROBABILITY

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks


# MISSING PARTs
# 1) send the distribution (mean and std) of the data if requested (for example, how the two classes are distrubuted over the age of the population (or any other feature))
# 2) send other useful data ? ((if available) feature importance, decision_path)
# ...

# training data -> expected to be with all the listed features (IN ORDER -> like in the data we have). It is ok, if there are missing values


# send and receive -> trigger keywords:
"""
	the message is generally composed by 2 elements ("nameOfTheMethod", "data")
	train + data.  // not used
	predict + data
"""

class LocalModelCommunication (LocalModel):
	
	# local model functions
	# train
	# predict 
   
	
	# initialize the local model with the training data

	def __init__(self, caller = "none" , data = "none", target_name = "AFclass" , model_name = "dt4",random_state = 12345678, imputation_strategy = 'mice',balance_strategy = 'SMOTE', local_models = []):
		LocalModel.__init__(self, data = data, target_name = target_name, model_name = model_name, random_state = random_state, imputation_strategy = imputation_strategy, balance_strategy = balance_strategy)
		#after the model is trained -> activate communication
		'''
		self.host = "127.0.0.1" ## server ip
		self.port = 33000 ## server port
		self.BUFSIZ = 102400000
		self.ADDR = (self.host, self.port)
		self.client_socket = socket(AF_INET, SOCK_STREAM) # create a socket connection
		self.client_socket.connect(self.ADDR) # try to connect to the given addess
		self.receive_thread = Thread(target=self.receive) # initialize a thread that waits for messages
		self.receive_thread.start()
		self.sendMessage("initialize connection")
		'''
		self.chunkNumber = 0 # used to send multiple chunks for prediction
		self.numberOfChunks = 0
		self.predictedData = []
		self.splits = [] # list of prediciton splits
		self.connectedToCentral = False
		self.caller = caller

	def connectToCentral(self):
		if self.connectedToCentral == False:
			try :
				self.host = "127.0.0.1" ## server ip
				self.port = 33000 ## server port
				self.BUFSIZ = 102400000
				self.ADDR = (self.host, self.port)
				self.client_socket = socket(AF_INET, SOCK_STREAM) # create a socket connection
				try :
					self.client_socket.connect(self.ADDR) # try to connect to the given addess
					self.receive_thread = Thread(target=self.receive) # initialize a thread that waits for messages
					self.receive_thread.start()
					self.sendMessage(self.localModelType)
					self.connectedToCentral = True
				except:
					print ("error connection")

			except :
				print ("unable to connect to the central server")
		else :
			return True
		return False


	def receive(self):
		"""Handles receiving of messages."""
		# infinite loop cycle
		while True:
			try:
				self.msg = self.client_socket.recv(self.BUFSIZ).decode("utf8")
				#msg_list.insert(tkinter.END, msg)
				print ("message recevied")
				print ("MESSAGE : ", self.msg)
				try :
					self.msg = json.loads(self.msg)
					if self.msg[0] == "train":
						print ("train set received")
						self.data_received = pd.read_json(self.msg[1])
						self.sendData(self.predict(self.data_received), "trainset predicted")

					if self.msg[0] == "training received":
						self.sendNextChunkOfTrainingData()


					if self.msg[0] == "predict":
						self.data_received = pd.read_json(self.msg[1])
						# perform prediction
						print ("PREDICYION 1")
						self.respondToPredictRequest(self.data_received)

					if self.msg[0] == "result":
						result = pd.read_json(self.msg[1])
						print ("RESULT 1")
						#print (result)
						self.predictionResultsReceived(result)


				except:
					print ("JSON ERROR")
			except OSError:  # Possibly client has left the chat.
				break
	
	'''
	def sendData(self, msg = "Message TEST -> sendData", event=None):  # event is passed by binders.
		"""Handles sending of messages."""
		print ("SENDING DATA")
		self.client_socket.sendall(bytes(msg, "utf8"))
		# quit connection
		if msg == "{quit}":
			self.client_socket.close()
	'''

	def sendData(self, data, label):
		
		to_send = data.to_json()
		message = json.dumps((label, to_send))
		self.client_socket.sendall(bytes(message, "utf8"))
		print ("message sent")

	def sendMessage(self, msg = "Message"):
		self.client_socket.sendall(bytes(msg, "utf8"))


	def respondToPredictRequest (self, data):
		print ("PREDICYION 2")
		'''
		prediction = self.predict(data).to_json()
		message = json.dumps(("predict" , prediction))
		self.sendData(message)
		'''
		self.sendData(self.predict(data), "predict")
		print ("PREDICYION 3")


	def predictionResultsReceived(self, data):
		# display data?
		# in the result we have also the predictions of the other hospital
		# the final prediction is in result[finalPrediction]
		data.sort_index(inplace = True)
		temp  = data["finalPrediction"]
		self.predictedData.append(temp)
		if self.chunkNumber < self.numberOfChunks:
			self.requestNewPrediction(self.splits[self.chunkNumber])
			self.chunkNumber += 1
		else :
			self.splits = []
			self.numberOfChunks = 0
			self.chunkNumber = 0
			print ("self predictedData ")
			print (self.predictedData)
			b = pd.DataFrame()
			for i in self.predictedData:
				b = pd.concat([b,i])
			self.predictedData = b.rename(index=str, columns={0: "finalPrediction"})
			## HERE WE HAVE RECEIVED THE PREDICTED DATA -> call other methods to display or analize
			if (str(self.caller) != "none"):
				## call the function you want to be triggered here!
				## self.caller.<FUNCTION> # the self.predictedData is the dataframe with the results
				print ("DATA PREDICTION FIANL")
				print (self.predictedData)


	def sendNextChunkOfTrainingData(self):
		if self.chunkNumber < self.numberOfChunks:
			self.sendData(self.splits[self.chunkNumber], "train") # send next chunk of data
			self.chunkNumber += 1
		else :
			# reset
			self.splits = []
			self.numberOfChunks = 0
			self.chunkNumber = 0
			print ("all the data has been sent")
			if (str(self.caller) != "none"):
				## call the function you want to be triggered here!
				## self.caller.<FUNCTION> # no specific paramenter
				print ("all the data has been sent")

	def requestNewPrediction(self,data):
		self.sendData(data, "requested_prediction")

	def requestPrediction(self, data):
		if self.connectedToCentral == True:
			print ("request prediction to central model")
			self.predictedData = [] # reset the received data
			self.splits = self.splitDataframe(data) # get the chunks of data
			self.numberOfChunks = len(self.splits)
			self.requestNewPrediction(self.splits[0])
			self.chunkNumber += 1
		else :
			print ("prediction on local model")
			self.predictedData = self.predict(self.data)
			print (self.predictedData)


	def sendDataForCentralTraining (self, data) :
		if self.connectedToCentral == True:
			data = data.reset_index().drop('index', axis = 1)
			self.splits = self.splitDataframe(data) # get the chunks of data
			self.numberOfChunks = len(self.splits)
			self.sendData(self.splits[0], "train")
			self.chunkNumber += 1
		else :
			print ("NOT connected -> ")





