# Multidisciplinary Project
* This is a multidisciplinary project carried out by a team of 5. The ECG data and patient information gathered from several European hospitals are provided. Our task is to improve the diagnose precision of atrial fibrillation.
* In this project, considering the lack of data and the data isolation of hospitals, we creatively used a second-layer stacking model, which is not only adaptive to the current existent models of some hospitals but also improved the prediction accuracy. The stacking model takes the outputs of the first-layer models as the input of the second-layer model. Our team used classifiers such as AdaBoost and RandomForest in the first layer, and logistic regression in the second layer. Patient data will be evaluated by its integrated model, then the result will be returned to the requested hospital as a reference for the doctor, and the final diagnoses made by a doctor will be inserted in the dataset for daily stack model training.
* I took charge of data cleaning and model building. The dataset is severely non-uniformed in units and unbalanced on gender and age. I used MICE impute and SMOTE over-sampling method for missing value handling.

# Note

* The original repository was created in 2018 . Because it had been set as private, I copied the whole project code in my own Github account for representation use.
* The work I have done includes data cleaning and model building.
* All the data preparations are in [source/](https://github.com/hades208002/mdp-project/tree/master/source),which includes many experiments tried to clean data with different impute and sampling models. The summary code is in [source/main.ipynb](https://github.com/hades208002/mdp-project/blob/master/source/main.ipynb).
* We assume that a presentation with 3 independent hospitals will be shown, so three local hospital models are trained.
* Our diagnose system is in [client_server_test/](https://github.com/hades208002/mdp-project/tree/master/client_server_test),see [client_server_test/how to use](https://github.com/hades208002/mdp-project/blob/master/client_server_test/how%20to%20use) to know how to implement this system.


## Python Environment 

1. Install Miniconda or Anaconda following these steps from [here](https://conda.io/docs/user-guide/install/index.html)
2. Create your environment

```bash
conda create --name mdp python=3.6
```

3. Activate your environment

```bash
source activate mdp
```

4. Install libraries

```bash
cd PROJECT_FOLDER_PATH
source activate mdp
pip install -r requirements.txt
```

4. [Optional] Test Jupyter Notebook 

```bash
jupyter notebook
```

5. Deactivate your environment

```bash
source deactivate mdp
```

## Helpful Links

### Atrial Fibrillation

- [Publications from AF Classification from a short single lead ECG recording](https://physionet.org/challenge/2017/papers/)
- [Atrial Fibrillation Detection Using Boosting and Stacking Ensemble](http://prucka.com/2017CinC/pdf/068-247.pdf)
- [Atrial fibrillation detection with a deep probabilistic model](https://medium.com/data-analysis-center/atrial-fibrillation-detection-with-a-deep-probabilistic-model-1239f69eff6c)

### Dealing With Imbalanced Datasets

- [Scikit-learn contrib library  `imbalanced-learn`](http://contrib.scikit-learn.org/imbalanced-learn/stable/)
- [Tutorial about `imbalaced-learn` ](https://blog.dominodatalab.com/imbalanced-datasets/)

### Titanic Dataset

- [Data Analysis on Titanic Dataset](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
- [Ensembling/Stacking in Python on Titanic Dataset](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

### Stacking/ Ensembling

- [Basic Stacking Guide from Kaggle No Free Hunch Blog](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
- [Stacking Made Easy: StackNet](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)
- [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)

### Visualization

- [Data Visualization Basics](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)
- [Complete Data Visualization Course from Kaggle](https://www.kaggle.com/learn/data-visualisation)
- [Easy Data Visualization Tutorial](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)

### Others

- [Github Guide](https://guides.github.com)
- [Dropbox Folder shared by Daiele Loiacono](https://www.dropbox.com/l/scl/AAD0auRNoQTeoxAvNlNIBGlW_fiGsavO2Zk)
