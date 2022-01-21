# The libraries that we will need throughout the project
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

# for making output full : 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Let's read our data
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()

df.columns = [col.upper() for col in df.columns]
y = df["CHURN"]
X = df.drop(['CUSTOMERID',"CHURN"], axis=1)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# df gibi birlesdirelim yeniden
df = pd.concat([X, y], axis=1)

df.tail()
df.shape
df.info()
df.columns
df.isnull().values.any() # Means there is no 'NA' in the data
df.isnull().sum() # How many 'NA' do we have in each column
df.isnull().values.sum() # How many 'NA' do we have in total in the whole data

# Let's see whole picture of data with that function
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# EDA
# This function will help us to determine numeric, categoric and seems as a catgorical but due to this column has a lot of class we consider it as cardinal
# columns 
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
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
# Let's use that functon 
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols # 'TENURE' 'MONTHLYCHARGES' CAN BE CATEGORICAL AS WELL by looking relation with target column
cat_but_car # We have to change the data type of 'TOTALCHARGES'
df['TOTALCHARGES'].nunique()

' ' in df['TOTALCHARGES'].unique()
df["TOTALCHARGES"] = df["TOTALCHARGES"].str.replace(' ', '0').astype(float)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.info()
df.head()

# Analysing Categorical Columns

# Let's investigate categoric columns
def cat_summary(dataframe, col_name, ratio=True, plot=False):
    if ratio:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts()}))
        print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col, plot = True)
# It seems we have 'imbalance data' problem in here as well

# Analysing Numerical Columns

df.describe().T # Ignore 'SENOIRCITIZEN'
# Comments:
    # It seems we have outliers in "TOTALCHARGES" column since median and mean are too diffirent from each other

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:
    num_summary(df, col, True)
    
# Analysis of Target Variable 

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

df.head()
label_encoder(df, 'CHURN')
df.head()

# Let's analyse categoric columns in terms of target column 'CHURN'
def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        'COUNT': dataframe.groupby(categorical_col)[target].count()}),
                         end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df, 'CHURN', col) 

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "CHURN", col)
    
# Outlier Detection Part

# Do we have any outliers in our numerical columns ?
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Since all columns return 'False' that means I do not have any outlier in my numeric columns
for col in num_cols:
    print(col, check_outlier(df, col))
    
df.isnull().values.any()
df.head()

#feature_extract
df["NEW_CHARGES_RATE"]=df["TOTALCHARGES"]/df["MONTHLYCHARGES"]
df["NEW_TOTALCHARGES/TENURE"]=df["TOTALCHARGES"]*df["TENURE"]

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
binary_cols 

for col in binary_cols:
    label_encoder(df, col)
df[binary_cols].head() # Ready

cat_cols.remove('CHURN')

cat_cols

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "CHURN", cat_cols) 

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe



df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "CHURN", cat_cols) 

df["NEW_TENURE_LEVEL"] = pd.qcut(df['TENURE'], 5, labels=[ 'E', 'D', 'C', 'B', 'A'])
target_summary_with_cat(df, 'CHURN', "NEW_TENURE_LEVEL")

df["NEW_TOTALCHARGES_LEVEL"] = pd.qcut(df['TOTALCHARGES'], 3, labels=[ 'C', 'B', 'A'])
target_summary_with_cat(df, 'CHURN', "NEW_TOTALCHARGES_LEVEL")

df["NEW_MONTHLYCHARGES_LEVEL"] = pd.qcut(df['MONTHLYCHARGES'], 3, labels=[ 'C', 'B', 'A'])
target_summary_with_cat(df, 'CHURN', "NEW_MONTHLYCHARGES_LEVEL")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
    
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.shape

df.drop('CUSTOMERID', axis=1, inplace=True)
df.head()

# Modelling
y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)

X, y = oversample.fit_resample(X, y)
# df gibi birlesdirelim yeniden
df = pd.concat([X, y], axis=1)



rf_model = RandomForestClassifier(random_state=46)

cv_results = cross_validate(rf_model,  # Modelimizi verelim
                            X, y,  # Bagimli ve bagimsiz deyiskenler
                            cv=10,  # k - sayisi, yani kac kisima bolsun verilen datayi
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]) # Hangi metriclere bakmak isdiyorsak onlari burda belirtmemi

cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


