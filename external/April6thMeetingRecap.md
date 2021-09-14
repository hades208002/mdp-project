
# 6th of April Meeting recap

## considerations & questions about the data
- How important is the ECG? we can easily identify two small groups of patients: 
1) patients with IPG = 1 -> these patients have some missing features. This is due to the fact that IPG = 1. However, only 2 out of 12 patients also miss the ECG data. 
2) there are about 50 patients that miss the ECG. in our dataset ECG accounts for more than 40 feautures. How should we deal with them? if ECG is not so important, maybe we can keep them. If we remove them we drastically remove the percentage of missing values, since the patients with less than 10 feauturs set to NaN are more than 1000. If we decide to remove these 50 patients the "new" critical features become : QRS and Pfwhm (in all its forms, e.g. Pfwhm_ECG_V1).

- How doctors nowadays distiguish the two classes?
1) By plotting the distributions of the features values for each class, we can notice (by eye) that they almost overlap (true for all the features). We should perfrom some deeper analysis on this point to have a clearer understanting if indeed there are differences.
2) By pair plotting w.r.t to the 2 classes the first 10 PCAs (which account for 73% of the total variance) we cannot easily identify "possible" clusters. Indeed in these 10 dimentions the two classes seems (at eye) to be indistinguishable.  
3) By simply applying a Classification method to the entire dataset (the missing values are filled with mean, while the PCneg feature is filled with 0) we get a result that is comparable to "choosing always the majority class". Notice that Persistent class size is about 1/3 of the other (on a total of 1100 samples). -> need to recheck using a dataset that contain the same amount of samples for each class.

## Concept Presentation Guidelines
- present some scenarios (at least 3 : one for each end user -> database manager, doctor , patient):
1) Database manager. Need for a package that helps him to insert the data in the system. We might present a set of features to be compared to the ones recorded in the local hospital in order to help the datamanager during the input phase. We need to think how to manage possible missining features, and how to introduce new features (how to update the previous dataset, if it is possible). Once the data has been inserted, the model is automatically built. At this point, the datamanager can deploy the system on the hospital local machines. Finally doctors can perform request to the systems and the system provides a responce.
2) Doctor. Interested in the result and in its visualization. When the doctor sends a request to the system the local model (of the hospital) computes the analysis, this analysis plus the data of the patient is sent to our server. Our server sends the data to the other hospitals in order to be analyzed by their local models. Our model (the second layer model /metamodel) collects the responce of each of those hospitals and combines them in the final prediction. finally this prediction is returned to the doctor that can visualize it.
3) Patient. interested at most in the visualization. wants to know what is good, what is bad, and how he locates in the population distribution (both of healty and unhealty people) (possibly, without too many specific details).

- Describe the concept:
1) illustrate the entire model starting from the scenarios   
2) remarks on the privacy:  we are adopting the most conservative approach (we need to "protect" "only" the old data available (already) in the hospitals; new patients must allow us to share their data in order to be tested by the local models. Here we can propose some possibilities: A.1) the patient might allow us to store the data and use the data to train new models. A.2) the patient might just want to use the model to compute the classification but not to store any data. A.3) ... ).
3) considerations about the visualization. in order to provide more informative results to the doctors (and patients) we need some additional data about the population distribution. Are we allowed to send this data (from each local hospital to our server and finally to the doctor)? Maybe we can use the local data of the hospital in  which the doctor is working.
4) considerations about white / black box modeling. if we want to provide useful informations allegated to the responce we might need to adopt a white box modeling apporach.

## Anything Else? 
### feel free to add / modify anything! 





 
