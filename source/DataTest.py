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


class dataTest:
    
    ## to be used if you want to test and train on a very specific dataset
    ## after this initialization use -> train() and predict()
    def __init__(self, X, target,test_size = 0.33, random_state = 12345678, k_fold = 10):
        
        
        
        # create test and training sets
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X, target, test_size=test_size, random_state=random_state)
        trainData = pd.concat([train_x,train_y], axis = 1)
        test_y = test_y.ravel()


        
        
        self.x_train = train_x
        self.y_train = train_y
        self.x_test = test_x
        self.y_test = test_y
        self.random_state = random_state
        self.models = [] ## list of the models
        self.models_definition(self.random_state)
        
        self.cv_x = X
        self.cv_y = target
        self.k_fold = k_fold
 

    def crossValidation(self):
        # cross validation
        print ("begin cross validation")
        evaluation = []
        counter = 1
        numberOfModels = len(self.models)

        best = ("BEST", 0, 0)

        for (name,i) in self.models:
            print (round(counter / numberOfModels,3), " is complete \t" )
            e = model_selection.cross_val_score(i, self.cv_x, self.cv_y, cv=StratifiedKFold(n_splits=self.k_fold,random_state=self.random_state,shuffle=True))
            avg = round(np.average(e),4) * 100
            std = round(np.std(e),4) * 100
            evaluation.append ((name, avg , std))
            counter = counter + 1
            if best[1] < avg :
                best = ("BEST : " + name, avg, std)
        print ("end cross validation")
        evaluation.append(best)
        evaluation.sort(key = lambda tup: tup[1], reverse = True)
        df_cv = pd.DataFrame (evaluation)
        return df_cv

    def models_definition(self,random_state):
        
        ## here we can tune the paramenters of the models
       

        self.models.append(("ada1",AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state = self.random_state),algorithm="SAMME", n_estimators=200)))
        self.models.append(("ada2",AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, random_state = self.random_state),algorithm="SAMME", n_estimators=200)))
        self.models.append(("ada3",AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state = self.random_state),algorithm="SAMME", n_estimators=100)))
        self.models.append(("ada4",AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, random_state = self.random_state),algorithm="SAMME", n_estimators=300)))
        self.models.append(("ada5",AdaBoostClassifier(DecisionTreeClassifier(max_depth=20, random_state = self.random_state),algorithm="SAMME", n_estimators=100)))
        self.models.append(("ada6",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))
        self.models.append(("ada7",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))
        self.models.append(("ada8",AdaBoostClassifier(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False))))

        #self.model.append(RadiusNeighborsClassifier(radius=10.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'))
       
        self.models.append(("ridge1", RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=self.random_state)))
        
        paramsGB1 = {'n_estimators': 120, 'max_depth': 3, 'subsample': 0.5,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': self.random_state}
        paramsGB2 = {'n_estimators': 120, 'max_depth': 6, 'subsample': 0.5,'learning_rate': 0.05, 'min_samples_leaf': 1, 'random_state': self.random_state}    
        paramsGB3 = {'n_estimators': 60, 'max_depth': 15, 'subsample': 0.5,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': self.random_state}
        paramsGB4 = {'n_estimators': 320, 'max_depth': 10, 'subsample': 0.5,'learning_rate': 0.005, 'min_samples_leaf': 1, 'random_state': self.random_state}
        self.models.append(("gb1",GradientBoostingClassifier(**paramsGB1)))
        self.models.append(("gb2",GradientBoostingClassifier(**paramsGB2)))
        self.models.append(("gb3",GradientBoostingClassifier(**paramsGB3)))
        self.models.append(("gb4",GradientBoostingClassifier(**paramsGB4)))

        self.models.append(("dt1",DecisionTreeClassifier(random_state=self.random_state)))
        self.models.append(("dt2",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
        self.models.append(("dt3",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
        self.models.append(("dt4",DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))
        self.models.append(("dt5",DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.random_state, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))


        self.models.append(("rf1",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
        self.models.append(("rf2",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
        self.models.append(("rf3",RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False)))
        
        self.models.append(("ld1",LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)))
       
        self.models.append(("lr1",LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=self.random_state, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)))
       
        self.models.append(("knn1",KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
        self.models.append(("knn2",KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
        self.models.append(("knn3",KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
        self.models.append(("knn4",KNeighborsClassifier(n_neighbors=20, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
        self.models.append(("knn5",KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)))
       
        self.models.append(("nb1",GaussianNB()))
 
        self.models.append(("et1",ExtraTreesClassifier(n_estimators=50, random_state=self.random_state)))     
        self.models.append(("et2",ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)))      
        self.models.append(("et3",ExtraTreesClassifier(n_estimators=200, random_state=self.random_state)))       

        self.models.append(("bag1",BaggingClassifier(base_estimator=None, n_estimators=5, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag2",BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag3",BaggingClassifier(base_estimator=None, n_estimators=20, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag4",BaggingClassifier(base_estimator=None, n_estimators=50, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag5",BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag6",BaggingClassifier(base_estimator=None, n_estimators=150, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag7",BaggingClassifier(base_estimator=None, n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag8",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag9",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=5, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag10",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=10, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))
        self.models.append(("bag11",BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=20, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=self.random_state, verbose=0, warm_start=False), n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=self.random_state, verbose=0)))

        ## add other models ...
        
               
    

    def train(self, x_train = 'self_train_x', y_train = 'self_train_y'):
        if x_train == 'self_train_x' and y_train == 'self_train_y':
            x_train = self.x_train
            y_train = self.y_train
        
        print ("START TRAINING")
        for (n,i) in self.models:
            i.fit(x_train,y_train)     
        print ("END TRAINING")
    
    
    def statistics(self,predicted_y, test_y):
        countPerTrue = 0
        countParTrue = 0
        countPerFalse = 0
        countParFalse = 0
        result = []
        for i in range(test_y.size):
            #print (test_y[i],predicted_y[i],"\n")
            if (test_y[i] == predicted_y[i]) and (test_y[i]== 1):
                countPerTrue += 1
            if (test_y[i] == predicted_y[i]) and (test_y[i]== 0):
                countParTrue += 1
            if (test_y[i] != predicted_y[i]) and (test_y[i]== 1):
                countParFalse += 1
            if (test_y[i] != predicted_y[i]) and (test_y[i]== 0):
                countPerFalse += 1
            #print (Y_bal_1_array[i],Y_P_bal_1[i])
        result.append(countPerTrue)
        result.append(countParTrue)
        result.append(countPerFalse)
        result.append(countParFalse)
        result.append((countPerTrue + countParTrue)/test_y.size) #ACCURACY
        result.append(countPerTrue/ (countPerTrue+countPerFalse)) #PRECISION
        result.append(countPerTrue/ (countPerTrue+countParFalse)) #RECALL
        
        return result
    
    def predict(self, x_test = "self_test_x", test_y = "self_test_y"):
        if x_test == "self_test_x" and test_y == "self_test_y":
            x_test = self.x_test
            test_y = self.y_test
        
        prediction = []
        
        for (n,i) in self.models:
            prediction.append((n,self.statistics(i.predict(x_test).ravel(), test_y)))
        
        df_prediction = pd.DataFrame(prediction)
        return df_prediction
    