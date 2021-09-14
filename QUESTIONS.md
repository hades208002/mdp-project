#QUESTIONS with Answer

## Considerations & questions about the data
1)  How important is the ECG? we can easily identify two small groups of patients:
- patients with IPG = 1 -> these patients have some missing features. This is due to the fact that IPG = 1. However, only 2 out of 12 patients also miss the ECG data.
- there are about 50 patients that miss the ECG. in our dataset ECG accounts for more than 40 feautures. How should we deal with them? if ECG is not so important, maybe we can keep them. If we remove them we drastically remove the percentage of missing values, since the patients with less than 10 feauturs set to NaN are more than 1000. If we decide to remove these 50 patients the "new" critical features become : QRS and Pfwhm (in all its forms, e.g. Pfwhm_ECG_V1).

2) How doctors nowadays distiguish the two classes?
- By plotting the distributions of the features values for each class, we can notice (by eye) that they almost overlap (true for all the features). We should perfrom some deeper analysis on this point to have a clearer understanting if indeed there are differences.
-  By pair plotting w.r.t to the 2 classes the first 10 PCAs (which account for 73% of the total variance) we cannot easily identify "possible" clusters. Indeed in these 10 dimentions the two classes seems (at eye) to be indistinguishable.
-  By simply applying a Classification method to the entire dataset (the missing values are filled with mean, while the PCneg feature is filled with 0) we get a result that is comparable to "choosing always the majority class". Notice that Persistent class size is about 1/3 of the other (on a total of 1100 samples). -> need to recheck using a dataset that contain the same amount of samples for each class

3) Is removing the IPG = 1 patients really a good idea? what if we need to classify a new patient with IPG = 1 ?

4) What's the most important class? Should we focus more on correctly classify the persistant or the paroxysmal class? (True Positive, False Positive, True Negative, False Negative)


## Questions about Data Mining

1) Consider the PCA Analysis (in a 2_class_classification problem). Suppose that we can describe with 10 principal components (over 100) the 90 % of the variance. If by pair_plotting all these 10 PC we cannot distinguish ("quite clearly") the two classes (in any of the pair-plot), can we "say" that the problem of classification is not easy? 



## Questions about Local Models

1) How should we train the local models? suppose that in a local hospital we have 90% Parox and 10% Pers. Should we try to balance the classes ? or train with all the data (and hope in an accuracy bigger than 90% ?) How to combine such unbalanced results?

2) in the case the test smaple has some missing value, cna we use the local data of the hospital to impute?




