# Main goal 

Build a model to predict which stage of  disease the patient is in.

# Project goal
1) build simple models by local dataset of each hospital
   methods mentioned : decision tree, deep learning, neural network, support vector machine, random forest 
2) build a new model which merge the single models by trying two or three different assembling methods
  methods mentioned : bagging ,boosting, stacking
  
# Problems to solve

1) each hospital has their own database that could contain different features
2) how to deal with missing data
3) same features may have different names in different local dataset
4) same features in same name may have different unit or measurement due to the difference among countries
5) difficulty to merge several models in assembling (for stacking is the outputs from the first tier may be different 
in data points of view)
6) the result of the model should be easy to understand by doctors

# State of the art

1) search the keywords
to get what people typically do with application like this 
     keywords mentioned : distributed dataset , stacking 
2) build a table about the paper we read, each  paper should has a description, the techniques proposed (boosting /bagging ...) ,
the field of application ( data stream/financial data processing ...) and other attributes if we think they are necessary.
