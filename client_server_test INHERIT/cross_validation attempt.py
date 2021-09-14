from sklearn.model_selection import StratifiedKFold


# data, models
d, models, features, 

skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(X,y):
    print (pd.DataFrame(train).shape, pd.DataFrame(test).shape)
    train_df = d.iloc[train]
    test_df = d.iloc[test]
    
    # balance train_df
     X_resampled, y_resampled = SMOTE().fit_sample(X, y_new)