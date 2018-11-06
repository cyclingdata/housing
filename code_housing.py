import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
import xgboost as xgb
dataset = pd.read_csv("train.csv")

scaler = StandardScaler()
submission = pd.read_csv("test.csv")

dataset['is_paved'] = np.where(dataset['Street']=="Paved",1,0)
submission['is_paved'] = np.where(submission['Street']=="Paved",1,0)
y = np.log(dataset[["SalePrice"]].values.ravel())

numeric_columns = ["LotFrontage","LotArea",
                   "OverallQual","OverallCond",
                   "YearBuilt","YearRemodAdd",
                   "MasVnrArea","BsmtFinSF1",
                   "BsmtFinSF2","BsmtUnfSF",
                   "TotalBsmtSF","1stFlrSF","2ndFlrSF",
                   "LowQualFinSF","GrLivArea","BsmtFullBath",
                   "BsmtHalfBath","FullBath","HalfBath",
                   "BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd",
                   "Fireplaces","GarageYrBlt","GarageCars",
                   "GarageArea","WoodDeckSF","OpenPorchSF",
                   "EnclosedPorch","3SsnPorch",
                   "ScreenPorch","PoolArea","MiscVal",
                   "YrSold","is_paved"]

categorical_columns = ["MSSubClass","MSZoning","Street","Alley","LotShape",
                      "LandContour","Utilities","LotConfig","LandSlope",
                       "Neighborhood","Condition1","Condition2",
                      "BldgType","HouseStyle","RoofMatl","Exterior1st",
                      "Exterior2nd","MasVnrType","ExterQual","ExterCond",
                      "Foundation","BsmtCond","BsmtExposure","BsmtFinType1",
                       "BsmtFinType2","Heating",
                          "HeatingQC","CentralAir","Electrical",
                         "KitchenQual","Functional","FireplaceQu",
                         "GarageType","GarageFinish","GarageQual",
                          "GarageCond","PavedDrive","PoolQC",
                        "Fence","MiscFeature","MoSold","SaleType",
                         "SaleCondition","BsmtQual","RoofStyle"]
                         


# replace infinite values
for cat in list(dataset.columns.values):
    if dataset[cat].dtypes in ['float','int']:
        dataset[cat]= dataset[cat].astype("float")
        n = dataset.loc[dataset[cat] != np.inf, cat].max()
        dataset[cat].replace(np.inf,n+1,inplace=True)
        mean = dataset[cat].mean()
        dataset[cat]=dataset[cat].fillna(mean)
        if cat != "SalePrice":
            submission[cat] = submission[cat].astype(dataset[cat].dtypes)
            n = submission.loc[submission[cat] != np.inf, cat].max()
            submission[cat].replace(np.inf,n+1,inplace=True)
            mean = submission[cat].mean()
            submission[cat]=submission[cat].fillna(mean)
numerical_df = dataset[numeric_columns]
sub_df = submission[numeric_columns]
ids = submission[["Id"]]
dataset[categorical_columns] = dataset[categorical_columns].fillna("missing")
submission[categorical_columns] = submission[categorical_columns].fillna("missing")
dataset[categorical_columns] = dataset[categorical_columns].astype("category")
submission[categorical_columns] = submission[categorical_columns].astype("category")
dummies = pd.get_dummies(dataset[categorical_columns])
dummies_sub = pd.get_dummies(submission[categorical_columns])
#test = dataset.loc[(~np.isfinite(dataset)) & dataset.notnull()]
#dataset2 = dataset.drop(categorical_columns,axis=1)
dataset = pd.concat([numerical_df,dummies],axis=1)
submission = pd.concat([sub_df,dummies_sub],axis=1)
#dataset = dataset.drop("Id",axis=1)

X = dataset

data_dmatrix = xgb.DMatrix(data=X,label=y)

train_X,test_X,train_y,test_y = train_test_split(X,y)


get_numeric_data = FunctionTransformer(lambda x: x[numeric_columns],
                                       validate=False)
get_categorical_data= FunctionTransformer(lambda x:x[categorical_columns],
                                          validate=False)
numeric = Pipeline([('selector',get_numeric_data),
                    ('imp',SimpleImputer()),
                    ('scale',StandardScaler())])

#categoric = Pipeline([ 
 #   ('select',get_categorical_data),
  #  ('impute',SimpleImputer(missing_values=None,strategy="constant",fill_value="missing")),
 #   ('encoder',DataFrameMapper([(categorical_columns,LabelBinarizer())]))
 #   ])

#categoric.fit_transform(test_X,test_y)
#union = FeatureUnion(transformer_list=[('numeric',numeric),('cat',categoric)])
#pl = Pipeline([('union',union),('linreg',LinearRegression())])
#pl= Pipeline([('num',numeric),('rf',RandomForestRegressor())])
pl = xgb.XGBRegressor(objective="reg:linear")

gbm_param_grid = {
    'learning_rate': np.arange(0.05,1,0.05),
    'max_depth': np.arange(3, 10, 1),
    'n_estimators': np.arange(20, 200, 20),
    'gamma' : [0,0.01,0.1,1,10]
}

# run randomized search
n_iter_search = 70
random_search = RandomizedSearchCV(pl, param_distributions=gbm_param_grid,
                                   n_iter=n_iter_search, cv=5)


random_search.fit(train_X,train_y)

print(random_search.score(test_X,test_y))

prediction = random_search.predict(test_X)


plt.plot(test_y,prediction,'ro')
plt.show()

for var in list(X.columns.values):
    if var not in list(submission.columns.values):
        submission[var] = 0
for var in list(submission.columns.values):
    if var not in list(X.columns.values):
        submission=submission.drop(var,axis=1)
submission = submission[X.columns]
pred_sub = np.exp(random_search.predict(submission))
res_submit = pd.concat([ids,pd.DataFrame(pred_sub,index=submission.index,columns=["SalePrice"])],
                        axis=1)
res_submit["Id"] = res_submit["Id"].astype("int")
res_submit.to_csv("submission.csv",index=False)


