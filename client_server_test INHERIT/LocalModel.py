# Import all the useful libraries
import numpy as np
import pandas as pd
import fancyimpute
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold



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

from sklearn.externals import joblib


# MISSING PARTs
# 1) send the distribution (mean and std) of the data if requested (for example, how the two classes are distrubuted over the age of the population (or any other feature))
# 2) send other useful data ? ((if available) feature importance, decision_path)
# ...

# training data -> expected to be with all the listed features (IN ORDER -> like in the data we have). It is ok, if there are missing values

class LocalModel:
	
	# local model functions
	# train
	# predict 
   
	
	# initialize the local model with the training data

	def __init__(self, data = "none", target_name = "AFclass" , model_name = "ada4",random_state = 12345678, imputation_strategy = 'mice',balance_strategy = 'SMOTE'):
		# we train the model with all the available data 
		self.target_name = target_name ## it the name of the target column
		self.target = None ## it is the target vector
		self.data_lm = data ## it is the complete dataset -> will be modified
		self.original_data = data ## store a copy of the original data -> never modified
		self.X = None ## it is the data except the target
		self.features_lm = None ## available features
		self.imputation_strategy = imputation_strategy
		self.balance_strategy = balance_strategy
		# for cross-validation 
		self.model_accuracy = ""
		self.cv_x = None # data -> in principle equal to self.X
		self.cv_y = None # target -> in principle equal to self.target
		self.random_state = random_state # random state -> fixed for testing
		self.selected_model_name = model_name # name of the model -> default fixed
		self.selected_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, random_state = self.random_state),algorithm="SAMME", n_estimators=300)#DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)  ## default model
		self.models = [] ## list of all the available models
		self.important_features = pd.DataFrame([],columns = {"important features"})
		#if not isinstance(self, LocalModel):
		#    self.chosen_model(model_name) # select the chosen model -> otherwise use the default one
		#self.check1, self.check2, self.check3 = self.fixDataset(imputation_strategy = imputation_strategy, balance_strategy = balance_strategy) ## fix data set before training -> clean data (remove unused columns, convert categotical attributes into numerical), recover missing values (use a strategy to impute the missing values), balance the data set
		#if isinstance(self, LocalModel):
		# self.chooseModel_with_crossValidation()
		self.localModelType = "app" ## gui or app -> gui can only respond to predictions , app can only send prediction requests or send data to central model
		if not str(self.data_lm) == "none":
			self.localModelType = "gui"
			self.perfromLocalOperations()

	def perfromLocalOperations(self):
		self.fixDataset(imputation_strategy = self.imputation_strategy, balance_strategy = self.balance_strategy) ## fix data set before training -> clean data (remove unused columns, convert categotical attributes into numerical), recover missing values (use a strategy to impute the missing values), balance the data set
		#self.train()

	# initiate the models_definition

	def chooseModel_with_crossValidation_and_train(self):
		r = []
		if not str(self.data_lm) == "none":
			try :
				print ("TRY load model, " + self.selected_model_name)
				self.selected_model = joblib.load(self.selected_model_name + '.pkl')
				print ("model loaded")
				r = self.crossValidation(all_models = 0) ## just to get the accuracy and the std deviation
				print ("skip trainign -Z model loaded")
			except :
				self.models_definition(self.random_state)
				r = self.crossValidation(all_models = 1, k_fold = 10)
				found = 0
				for (n,i) in self.models: # n = name , i = model
					if n == r.iloc[0][0] and found == 0:
						found = 1
						self.selected_model = i
						self.selected_model_name = n
				if found == 0:
					self.selected_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
					self.selected_model_name = "dt4"
				self.train()
				joblib.dump(self.selected_model, self.selected_model_name + '.pkl') 
		else:
			print ("no data")
		self.getImportantFetures()
		print ("DONE Cross validation -> choosen model")
		print (self.important_features.shape)
		print (self.important_features)
		print ("shape")
		print (self.selected_model_name, self.selected_model)
		return r

	def getImportantFetures(self):
		a = []
		print (self.selected_model.feature_importances_)
		try:
			indices = np.argsort(self.selected_model.feature_importances_)[-10:]
			for i in indices :
				a.append(self.data_lm.columns[i])
			print ("important features here")
			print (self.important_features)
			print (" finish printing important features ")
		except :
			print("no features importance")
		a = pd.DataFrame(a,columns = {"important features"})
		self.important_features = a

	def updateFeatures(self,features):
		f = features["important features"].tolist()
		pos = 0
		res = self.important_features["important features"].tolist()
		try:
			for i in f:
				if i not in res:
					res.insert(pos , i)
					pos += 1
				else :
					oldPos = res.index(i)
					res.remove(i)
					res.insert(int((pos + oldPos) / 2) , i)
				pos += 1
			res = pd.DataFrame(res,columns = {"important features"})
			self.important_features = res
		except:
			print("error in update feature")
		return res


	def addData(self, data, target_name = "AFclass", localModelType = "gui"):
		# add data only if there is not other yet (in future -> possibility to concat to self.original_data)
		if str(self.data_lm) == "none":
			self.localModelType = "gui"
			self.data_lm = data
			self.target_name = target_name
			self.perfromLocalOperations()
			print ("data Added and fixed")
			return True
		print ("abort -> there is already data")
		return False


	def models_definition(self,random_state):
		
		## here we can tune the paramenters of the models
	   
		
		#self.models.append(("ada1",AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state = self.random_state),algorithm="SAMME", n_estimators=200)))
		#self.models.append(("ada2",AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, random_state = self.random_state),algorithm="SAMME", n_estimators=200)))
		#self.models.append(("ada3",AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state = self.random_state),algorithm="SAMME", n_estimators=100)))
		self.models.append(("ada4",AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, random_state = self.random_state),algorithm="SAMME", n_estimators=300)))
		#self.models.append(("ada5",AdaBoostClassifier(DecisionTreeClassifier(max_depth=20, random_state = self.random_state),algorithm="SAMME", n_estimators=100)))
		#self.models.append(("ada6",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))
		#self.models.append(("ada7",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))
		#self.models.append(("ada8",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))
		"""
		#self.model.append(RadiusNeighborsClassifier(radius=10.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'))
	   
		#self.models.append(("ridge1", RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=self.random_state)))
		
		#paramsGB1 = {'n_estimators': 120, 'max_depth': 3, 'subsample': 0.5,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': self.random_state}
		#paramsGB2 = {'n_estimators': 120, 'max_depth': 6, 'subsample': 0.5,'learning_rate': 0.05, 'min_samples_leaf': 1, 'random_state': self.random_state}    
		#paramsGB3 = {'n_estimators': 60, 'max_depth': 15, 'subsample': 0.5,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': self.random_state}
		paramsGB4 = {'n_estimators': 320, 'max_depth': 10, 'subsample': 0.5,'learning_rate': 0.005, 'min_samples_leaf': 1, 'random_state': self.random_state}
		#self.models.append(("gb1",GradientBoostingClassifier(**paramsGB1)))
		#self.models.append(("gb2",GradientBoostingClassifier(**paramsGB2)))
		#self.models.append(("gb3",GradientBoostingClassifier(**paramsGB3)))
		self.models.append(("gb4",GradientBoostingClassifier(**paramsGB4)))
		"""
		#self.models.append(("dt1",DecisionTreeClassifier(random_state=self.random_state)))
		#self.models.append(("dt2",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
		#self.models.append(("dt3",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
		self.models.append(("dt4",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
		#self.models.append(("dt5",DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))

		"""
		#self.models.append(("rf1",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
		self.models.append(("rf2",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
		#self.models.append(("rf3",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
		
		#self.models.append(("ld1",LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)))
	   
		#self.models.append(("lr1",LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=self.random_state, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)))
	   
		#self.models.append(("knn1",KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
		self.models.append(("knn2",KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
		#self.models.append(("knn3",KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
		#self.models.append(("knn4",KNeighborsClassifier(n_neighbors=20, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
		#self.models.append(("knn5",KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
		   
		#self.models.append(("nb1",GaussianNB()))
 		
		#self.models.append(("et1",ExtraTreesClassifier(n_estimators=50, random_state=self.random_state)))     
		#self.models.append(("et2",ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)))
		self.models.append(("et3",ExtraTreesClassifier(n_estimators=200, random_state=self.random_state)))       
		
		#self.models.append(("bag1",BaggingClassifier(base_estimator=None, n_estimators=5, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag2",BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag3",BaggingClassifier(base_estimator=None, n_estimators=20, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag4",BaggingClassifier(base_estimator=None, n_estimators=50, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag5",BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag6",BaggingClassifier(base_estimator=None, n_estimators=150, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag7",BaggingClassifier(base_estimator=None, n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		self.models.append(("bag8",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag9",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag10",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		#self.models.append(("bag11",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=20, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
		"""
		## add other models ...
		

	def chosen_model(self, name):
		# initialize the available models
		self.models_definition(self.random_state)
		found = 0
		for (n,i) in self.models: # n = name , i = model
			if n == name and found == 0:
				found = 1
				self.selected_model = i
				self.selected_model_name = name
		if found == 0 :
			# feel free to modify the model.. if another is better
			self.selected_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
			self.selected_model_name = "dt4"
		return


	## to choose the best model using cross validation
	## normally crossvalidate just the chosen model, if all_models = 1 -> crossvalidate all the models 
	def crossValidation(self, all_models = 0, k_fold = 10, random_state = 12345678):
		# cross validation
		if all_models == 1:
			print ("begin cross validation for all models")
			evaluation = []
			counter = 1
			numberOfModels = len(self.models)
			#best = ("BEST", 0, 0)
			for (name,i) in self.models:
				print (round(counter / numberOfModels,3), " is complete \t" )
				e = model_selection.cross_val_score(i, self.cv_x, self.cv_y, cv=StratifiedKFold(n_splits=k_fold,random_state=random_state,shuffle=True))
				avg = round(np.average(e),4) * 100
				std = round(np.std(e),4) * 100
				evaluation.append ((name, avg , std))
				counter = counter + 1
			evaluation.sort(key = lambda tup: tup[1], reverse = True)
			df_cv = pd.DataFrame (evaluation, columns=['Model','Accuracy','std'])
			self.model_accuracy = df_cv.iloc[0][1]
			print ("end cross validation")
			return df_cv
		else:
			e = model_selection.cross_val_score(self.selected_model, self.cv_x, self.cv_y, cv=StratifiedKFold(n_splits=k_fold,random_state=random_state,shuffle=True))
			t = pd.DataFrame([(self.selected_model_name, round(np.average(e),4) * 100 , round(np.std(e),4) * 100 )], columns=['Model','Accuracy','std'])
			return t
		return

	def showData(self, lines = 5, original_data = 0):
		if original_data == 1:
			print (self.original_data.head(lines))
		else :
			print(self.data_lm.head(lines))
	 

	# remove unused features, convert categorical attributes to numerical ones
	def cleanData(self):
		print ("START CLEANING")
		# re-start from the orginial data
		#self.data_lm = self.original_data
		if 'Soggetti' in self.data_lm.columns:
			self.data_lm = self.data_lm.drop('Soggetti', axis = 1)
		if 'PCneg' in self.data_lm.columns:
			self.data_lm = self.data_lm.drop('PCneg', axis = 1)
		if 'IPG' in self.data_lm.columns:
			self.data_lm = self.data_lm.drop('IPG', axis = 1)
		if 'sbjBeatConsidered' in self.data_lm.columns:
			self.data_lm = self.data_lm.drop('sbjBeatConsidered', axis=1)
		if 'numRRaveraged' in self.data_lm.columns:
			self.data_lm = self.data_lm.drop('numRRaveraged', axis=1)


		# convert categorical variables into numerical 
		if 'patsex' in self.data_lm.columns and ("männlich" in self.data_lm["patsex"].values or  "weiblich" in self.data_lm["patsex"].values):
			self.data_lm['patsex'] = self.data_lm['patsex'].map({'männlich' : 1, 'weiblich' : 0})

		if 'AFclass' in self.data_lm.columns and ("persistierend (>7 Tage, EKV)" in self.data_lm["AFclass"].values or  "paroxysmal" in self.data_lm["AFclass"].values):
			self.data_lm["AFclass"] = self.data_lm["AFclass"].map({'persistierend (>7 Tage, EKV)' : 1, 'paroxysmal' : 0}) 

		# extract features
		self.features_lm = self.data_lm.columns[self.data_lm.columns != self.target_name]
		self.X = self.data_lm[self.features_lm]
		self.target = self.data_lm[self.target_name]
		print ("END CLEANING")
	
	# clean the test data -> first drop unused data -> make it "compliant" to the features of the dataset 
	def cleanDataTest(self, test_x, features = "self_features"):
		print ("START TEST CLEANING")
		print("TEST_X : ", test_x.shape)
		print (test_x)

		# convert categorical variables into numerical 
		if 'patsex' in test_x.columns and ("männlich" in test_x["patsex"].values or  "weiblich" in test_x["patsex"].values):
			test_x['patsex'] = test_x['patsex'].map({'männlich' : 1, 'weiblich' : 0})

		if str(features) == "self_features":
			list_of_features = self.features_lm
		else :
			list_of_features = features

		# drop all the columns that are not present in the training dataset
		for i in test_x.columns:
			if i not in list_of_features:
				test_x = test_x.drop(i, axis = 1)

		# add columns that are not present in the test set
		for i in list_of_features:
			if i not in test_x.columns:
				test_x[i] = np.nan

		## REORDER the features
		test_x = test_x[list_of_features]
		print ("END TEST CLEANING")
		print("DATA shape : ", self.data_lm.shape)
		return test_x

	## data -> it is the dataset we want to 'recover'
	
	def imputeData(self, dataframe,imputation_strategy = 'knn', features = "self_features" ):
		try:   
			if imputation_strategy == 'knn':
				x_complete_a = fancyimpute.KNN(15).complete(dataframe)
		## feel free to add other imputation methods 
		# ... 
			else : ## default case -> MICE impute method
				mice = fancyimpute.MICE(n_imputations=100, impute_type='col', n_nearest_columns=5, init_fill_method = "mean")
				x_complete_a = mice.complete(dataframe)
		except:
			x_complete_a  = dataframe
		print ("x_incomplete shape : ",x_complete_a.shape )
		if str(features) == "self_features":
			f = self.features_lm
		else :
			f = features
		print ("FEATURESS : ",f.size, f )
		return pd.DataFrame(x_complete_a, columns = f)

	def recoverMissing(self, data = 'trainData', imputation_strategy = 'mice'):
		print ("START RecoverMissing VALUES")
		if str(data) == 'trainData':
			x_incomplete = self.data_lm[self.features_lm]	
		else:
			x_incomplete = data[self.features_lm]
		#print (x_incomplete)
		# create a united dataset -> suppose it is possile -> if we clean first ->> then it is possible

		if str(data) != 'trainData':

			united_df = pd.concat([x_incomplete, self.X])
			united_complete = self.imputeData(united_df, features = x_incomplete.columns)
			x_complete = united_complete.iloc[:x_incomplete.shape[0], :x_incomplete.shape[1]]
			#print ("united_complete shape : ",united_complete.shape )

		else :
			x_complete = self.imputeData(x_incomplete)
		'''
		try:   
			if imputation_strategy == 'knn':
				x_complete_a = fancyimpute.KNN(15).complete(x_incomplete)
		## feel free to add other imputation methods 
		# ... 
			else : ## default case -> MICE impute method
				mice = fancyimpute.MICE(n_imputations=100, impute_type='col', n_nearest_columns=5)
				x_complete_a = mice.complete(x_incomplete)
		except:
			x_complete_a  = x_incomplete
		
		x_complete = pd.DataFrame(x_complete_a, columns = self.features_lm)
		'''
		if str(data) == 'trainData':
			self.X = x_complete
		return x_complete
	
	def balanceDataSet(self, data = "trainData",target_name = "AFclass", balance_strategy = 'SMOTE'):
		if str(data) == "trainData":
			X = self.X
			y = self.data_lm[self.target_name].as_matrix()
			target_name = self.target_name
		else :
			X = data[data.columns[data.columns != target_name]]
			y = data[target_name].as_matrix()
		y_new = pd.DataFrame(y)
		y_new = y_new.rename(columns = {y_new.columns[0] : target_name})
		Data_complete = pd.concat([X,y_new], axis = 1)
		if balance_strategy == 'ADASYN':
			try:
				print ("Try ADASYN")
				X_resampled, y_resampled = ADASYN().fit_sample(X, y_new)
			except:
				print ("ADASYN FAILED -> used SMOTE")
				X_resampled, y_resampled = SMOTE().fit_sample(X, y_new)

			## feel free to add other balancing strategies
			# ...
		else : # default SMOTE
			X_resampled, y_resampled = SMOTE().fit_sample(X, y_new)



		X_final = pd.DataFrame(X_resampled, columns = self.features_lm)
		Y_final = pd.DataFrame(y_resampled)
		Y_final = Y_final.rename(columns = {Y_final.columns[0] : self.target_name})

		Data_final = pd.concat([X_final,Y_final], axis = 1)
		if str(data) == "trainData" :
			self.X = X_final
			self.target = Y_final
			self.cv_x = X_final
			self.cv_y = Y_final
			self.data_lm = Data_final
		return Data_final


	# clean the data, recover missing values, balance the dataset
	def fixDataset(self, imputation_strategy = 'mice', balance_strategy = 'SMOTE'):
		print ("begin fixing dataset")
		self.cleanData()
		check1 = self.data_lm.copy()
		self.recoverMissing(imputation_strategy = imputation_strategy)
		check2 = self.X.copy()
		self.balanceDataSet(balance_strategy = balance_strategy)
		check3 = self.data_lm.copy()
		print ("end fixing dataset")
		return (check1, check2, check3)

	# train the selected model
	def train(self):
		## use all the availble data -> we assume to know what is the best model -> otherwise use the crossvalidation function to choose a model
		print ("begin training")
		self.selected_model.fit(self.X, self.target)
		print ("end training")


	# predict using the trained model. x_test is a vector 
	# return the prediction for all values in the vector x_test, and all the other useful data (according to the selected_model used to predict)
	def predict(self, test):
		original_x = test
		train = test.copy()
		x_test = self.cleanDataTest(test_x = train)
		x_test = self.recoverMissing(data = x_test)
		result = x_test.copy()
		prediction = self.selected_model.predict(x_test)
		result['prediction'] = prediction
		#decision_path = None
		#features_importance = None
		
		if callable(hasattr(self.selected_model, "predict_proba" )):
			predict_proba_df = pd.DataFrame(self.selected_model.predict_proba(x_test), columns=self.selected_model.classes_)
			result['predict_proba_zero'] = predict_proba_df[predict_proba_df.columns[0]]
			result['predict_proba_uno'] = predict_proba_df[predict_proba_df.columns[1]]
		'''      
		if callable(hasattr(self.selected_model, "predict_log_proba" )):
			predict_log_proba_df = pd.DataFrame(self.selected_model.predict_log_proba(x_test), columns=self.selected_model.classes_)
			result['predict_log_proba_zero'] = predict_log_proba_df[predict_log_proba_df.columns[0]]
			result['predict_log_proba_uno'] = predict_log_proba_df[predict_log_proba_df.columns[1]]
		'''
		return pd.DataFrame(result)

	def splitDataframe (self, data, step = 20):
		splits = []
		i = 0
		n = data.shape[0]
		if n > step:
			while i < n:
				l = i + step
				temp = data.iloc[i: l, :]
				splits.append(temp)
				i += step
		else:
			splits.append(data)
		return splits
