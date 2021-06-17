import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
os.chdir("C:/Users/Ashutosh/Desktop/jobathon")


datatr = pd.read_csv(r"train_s3TEQDk.csv")
datats = pd.read_csv(r"test_mSzZ8RL.csv")
submission = pd.read_csv(r"sample_submission_eyYijxG.csv")

def modelling(train,test,submission):
    train = train.fillna("-99"); test = test.fillna("-99")
    train = train.drop("ID",axis=1); test= test.drop("ID",axis=1)
    y_train = train["Is_Lead"]; train = train.drop("Is_Lead",axis=1)
    category_col = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code',\
        'Credit_Product', 'Is_Active']
    for colum in category_col:
        train[colum] = train[colum].astype("category")
        test[colum] = test[colum].astype("category")
    category_col_index = np.where(train.dtypes == "category")[0]
    model = CatBoostClassifier(iterations=930,depth=5,
                               eval_metric="AUC",auto_class_weights="SqrtBalanced",
                               logging_level = "Silent")
    model.fit(train,y_train,cat_features=category_col_index)
    submission["Is_Lead"] = model.predict_proba(test)[:,1]
    submission.to_csv("output_file.csv",index=False)

modelling(datatr,datats,submission)
   