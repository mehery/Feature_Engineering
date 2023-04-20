import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv") 


def check_df(dataframe, head=5):
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("################# Duplicated ###################")
    print(dataframe.duplicated().sum())

check_df(df)


# First, we need to identify the numerical and categorical variables in the data.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    return cat_cols, num_cols, cat_but_car 


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        fig = plt.subplots(figsize=(6, 4))
        sns.distplot(df[col_name],
             kde=False,
             kde_kws={"color": "g", "alpha": 0.3, "linewidth": 5, "shade": True})
        plt.show(block=True)
        
        
        
# We are analyzing the numeric variables.

for col in num_cols:
    num_summary(df, col, plot = True)
    
    
# We are analyzing the categorical variables.

for col in cat_cols:
    fig =  plt.subplots(figsize=(7, 5))
    sns.countplot(x = df[col], data = df)
    plt.show(block = True)
    
    
# We are analyzing the target variable.

for col in num_cols:
    print(df.groupby('Outcome').agg({col: 'mean'}))
    fig = plt.subplots(figsize=(6, 4))
    sns.violinplot(x=df["Outcome"], y=df[col])
    plt.show(block=True)
    
    
    
# We are analyzing the outliers.

# To detect outliers, we need to set threshold values.
def outlier_thresholds(dataframe, col_name, q1=0.04, q3=0.96):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# We are checking the variables that have outliers.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    

for col in num_cols:
    print(col, check_outlier(df, col))
    
    
# We replace the outliers with the threshold values we determined.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
    
for col in num_cols:
    replace_with_thresholds(df, col)
    
check_outlier(df, num_cols)


### Local Outlier Factor(LOF)

# We generate our scores with LOF.
clf= LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

# We are examining the scores through a graph.
scores = pd.DataFrame(np.sort(df_scores))   
scores.plot(stacked=True, xlim=[0, 70], style='.-')
plt.show()

# This can be interpreted as the point where the rate of change starts to slow down after the 10th value.


# We set the 10th point as the threshold.
th= np.sort(df_scores)[10]

# We are looking at the outlier observation units that fall below this threshold.
df[df_scores < th]

# We remove these outlier observations from the dataset.
df = df[~(df_scores < th)]


# We are examining our correlation analysis.

corr = df[df.columns].corr()

sns.set(rc={"figure.figsize" : (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


# We are analyzing the missing values.

df.isnull().any()

# It seems no missing values. But at first we saw that there are meaningless values. 
# For example: Variables such as Glucose, Insulin, Skin Thickness and Blood Pressure cannot be 0.
# We will treat these variables as NaN.


# pregnancies can be 0, we leave it out
num_cols_miss = [col for col in num_cols if col not in "Pregnancies"]

# We set 0 values to NaN
for col in num_cols_miss:
    df[col] = np.where(df[col] == 0, np.NaN, df[col])
    
    
df.isnull().sum()


# We have identified missing values and we need to fill them in.

# "I will fill in these missing values with the medians of the variable groups that are most closely related."

# I am filling in the missing values in insulin using the median values of insulin with respect to glucose."
df["Glucose_qcut"]=pd.qcut(df['Glucose'], 5)
df.groupby("Glucose_qcut")["Insulin"].median()
df["Insulin"] = df["Insulin"].fillna(df.groupby("Glucose_qcut")["Insulin"].transform("median"))

# I am filling in the missing values in SkinThickness using the median values of SkinThickness with respect to BMI.
df["BMI_qcut"] = pd.qcut(df['BMI'], 5)
df.groupby("BMI_qcut")["SkinThickness"].median()
df["SkinThickness"] = df["SkinThickness"].fillna(df.groupby("BMI_qcut")["SkinThickness"].transform("median"))

# I am filling in the missing values in Pregnancies using the median values of Pregnancies with respect to Age.
df["Age_qcut"] = pd.qcut(df['Age'], 3)
df.groupby("Age_qcut")["Pregnancies"].median()
df["Pregnancies"] = df["Pregnancies"].fillna(df.groupby("Age_qcut")["Pregnancies"].transform("median"))

# I am filling in the missing values in BloodPressure using the median values of BloodPressure with respect to Age.
df.groupby("Age_qcut")["BloodPressure"].median()
df["BloodPressure"] = df["BloodPressure"].fillna(df.groupby("Age_qcut")["BloodPressure"].transform("median"))


# We are deleting the remaining small number of missing values and the newly created variables."

df.dropna(inplace=True)
df.drop(["Age_qcut", "BMI_qcut", "Glucose_qcut"], axis=1, inplace=True)

df.describe().T


# Now we are creating new variables.

df.loc[(df['Glucose'] < 140) & (df['BMI'] <= 25), 'NEW_GLU_BMI'] = 'thin_-'

df.loc[(df['Glucose'] < 140) & (df['BMI'] > 25) & (df['BMI'] < 30), 'NEW_GLU_BMI'] = 'fat_-'

df.loc[(df['Glucose'] < 140) & (df['BMI'] >= 30), 'NEW_GLU_BMI'] = 'obese_-'

df.loc[(df['Glucose'] >= 140) & (df['BMI'] <= 25), 'NEW_GLU_BMI'] = 'thin_+'

df.loc[(df['Glucose'] >= 140) & (df['BMI'] > 25) & (df['BMI'] < 30), 'NEW_GLU_BMI'] = 'fat_+'

df.loc[(df['Glucose'] >= 140) & (df['BMI'] >= 30), 'NEW_GLU_BMI'] = 'obese_+'


# Now we are creating new variables.

df.loc[(df['Age'] < 46) & (df['SkinThickness'] < 46), 'NEW_Age_SKIN'] = 'age<_skin<'

df.loc[(df['Age'] < 46) & (df['SkinThickness'] >= 46), 'NEW_Age_SKIN'] = 'age<_skin>'

df.loc[(df['Age'] >= 46) & (df['Age'] < 56) & (df['SkinThickness'] < 46), 'NEW_Age_SKIN'] = 'age--_skin<'

df.loc[(df['Age'] >= 46) & (df['Age'] < 56) & (df['SkinThickness'] >= 46), 'NEW_Age_SKIN'] = 'ages--_skin>'

df.loc[(df['Age'] >= 56) & (df['SkinThickness'] < 46), 'NEW_Age_SKIN'] = 'age>_skin<'

df.loc[(df['Age'] >= 56) & (df['SkinThickness'] >= 46), 'NEW_Age_SKIN'] = 'age>_skin>'


# Now we are creating new variables.

def calculate_glucose_age(row):
    if row['Age'] < 26:
        return row['Glucose']
    elif row['Age'] < 36:
        return row['Glucose'] * 1.2
    elif row['Age'] < 46:
        return row['Glucose'] * 1.4
    elif row['Age'] < 55:
        return row['Glucose'] * 1.6
    elif row['Age'] < 64:
        return row['Glucose'] * 1.8
    elif row['Age'] < 72:
        return row['Glucose'] * 2
    else:
        return row['Glucose'] * 2.4

df['Glucose_age'] = df.apply(calculate_glucose_age, axis=1)


# Now we are creating new variables.

df["FUNC_GLU"] = df["DiabetesPedigreeFunction"] * df["Glucose"]


df["SKIN_INS"] = df["SkinThickness"] * df["Insulin"]


df.head()


# We are performing the encoding process.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df= one_hot_encoder(df, ohe_cols)

df.head()


# We are performing standardization processes.

mms = MinMaxScaler()    
df[num_cols] = mms.fit_transform(df[num_cols])

df.head()


# We are applying our machine learning model.

from sklearn.ensemble import RandomForestClassifier

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=26)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy_score(y_pred, y_test)

# We can see the success rate of the model


# We observe the effects of the variables we added later on the dependent variable.

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

plot_importance(rf_model, X_train)










