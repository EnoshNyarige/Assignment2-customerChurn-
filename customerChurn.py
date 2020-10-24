#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
# pd.set_option("display.max_columns", 50)
import numpy as np
seed = 515
np.random.seed(seed)
import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


df = pd.read_csv('churn.csv')


# In[25]:


def display_plot(dataset, col_to_exclude, object_mode = True):
    """ 
     This function plots the count or distribution of each column in the dataframe based on specified inputs
     @Args
       df: pandas dataframe
       col_to_exclude: specific column to exclude from the plot, used for excluded key 
       object_mode: whether to plot on object data types or not (default: True)
       
     Return
       No object returned but visualized plot will return based on specified inputs
    """
    n = 0
    this = []
    
    if object_mode:
        nrows = 4
        ncols = 4
        width = 20
        height = 20
    
    else:
        nrows = 2
        ncols = 2
        width = 14
        height = 10
    
    
    for column in df.columns:
        if object_mode:
            if (df[column].dtypes == 'O') & (column != col_to_exclude):
                this.append(column)
                
                
        else:
            if (df[column].dtypes != 'O'):
                this.append(column)
     
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(width, height))
    for row in range(nrows):
        for col in range(ncols):
            if object_mode:
                sns.countplot(df[this[n]], ax=ax[row][col])
                
            else:
                sns.distplot(df[this[n]], ax = ax[row][col])
            
            ax[row,col].set_title("Column name: {}".format(this[n]))
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")
            n += 1

    plt.show();
    return None


# In[26]:


def convert_no_service (df):
    col_to_transform = []
    for col in df.columns:
        if (df[col].dtype == 'O') & (col != 'customerid'):
            if len(df[df[col].str.contains("No")][col].unique()) > 1:
                col_to_transform.append(col)
    
    print("Total column(s) to transform: {}".format(col_to_transform))
    for col in col_to_transform:
        df.loc[df[col].str.contains("No"), col] = 'No'
        
    return df


# In[27]:


df = convert_no_service(df)


# In[28]:


display_plot(df, 'customerID', object_mode = True)


# In[29]:


df.gender = df.gender.map(dict(Male=1, Female=0))


# In[30]:


def encode_yes_no (df, columns_to_encode):
    for col in columns_to_encode:
        df[col] = df[col].map(dict(Yes = 1, No = 0))
        
    return df
encode_columns = []
for col in df.columns:
    keep = np.sort(df[col].unique(), axis = None)
    
    if ("Yes" in keep) & ("No" in keep):
        encode_columns.append(col)

del keep
print("Encode Columns Yes/No: {}".format(encode_columns))


# In[31]:


df = encode_yes_no(df, encode_columns)
# display(df.head(5))


# In[32]:


df = pd.get_dummies(df, columns = ['InternetService', 'Contract', 'PaymentMethod'], prefix = ['ISP', 'contract', 'payment'])
# df.head(5)


# In[33]:


df.dropna(inplace = True)


# In[34]:


df


# In[35]:


# df2 = df[['customerID', 'tenure', 'MonthlyCharges', 'payment_Electronic check', 'contract_One year','TotalCharges', 'Churn']]


# In[36]:


df.head()


# In[37]:


dfd = pd.DataFrame(df)


# In[38]:


dfd = df.drop([
    'gender', 
    'SeniorCitizen', 
    'Partner', 
    'Dependents', 
    'PhoneService', 
    'MultipleLines', 
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport', 
    'StreamingTV', 
    'StreamingMovies', 
    'PaperlessBilling', 
    'ISP_DSL', 
    'ISP_Fiber optic', 
    'ISP_No', 
    'contract_Month-to-month', 
    'contract_Two year', 
    'payment_Bank transfer (automatic)', 
    'payment_Credit card (automatic)',
    'payment_Mailed check'], axis=1)


# In[39]:


dfd.head()


# In[40]:


# contract_One year, tenure, MonthlyCharges, payment_Electronic check, TotalCharges


# In[41]:


dfd1 = dfd[['customerID', 'contract_One year', 'tenure', 'MonthlyCharges', 'payment_Electronic check', 'TotalCharges', 'Churn']]


# In[42]:


dfd1


# In[43]:


dfd1.corr()['Churn'].sort_values(ascending=False)


# In[44]:


plt.figure(figsize=(16, 10))
dfd1.drop(['customerID'], axis=1, inplace=True)
corr = dfd1.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")


# In[45]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
import pickle
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer


# In[46]:


import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, auc


# In[47]:


X = dfd1.drop('Churn', axis = 1, inplace = False)
y = dfd1['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# y = y.astype('int')
# X = X.astype('int')


# In[52]:


import xgboost
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error


# In[53]:


if xgboost is not None: 
    best_xgb = xgboost.XGBClassifier(random_state=42)
    best_xgb.fit(X_train, y_train)
    y_pred = best_xgb.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    print("Validation MSE:", val_error)


# In[54]:


best_xgb = XGBClassifier(learning_rate = 1, max_depth = 4, n_estimators = 20, n_jobs = 4)

best_xgb.fit(
    X_train, y_train, 
    eval_set = [(X_train, y_train), (X_test, y_test)], 
    early_stopping_rounds = 20)

y_pred = best_xgb.predict(X_test)
val_error = mean_squared_error(y_test, y_pred)
print("\nValidation MSE:", val_error)


# In[55]:


print ("XGBOOST Classifier\n")

clf = XGBClassifier(n_jobs = 11)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

print ("Accuracy Score: ", score, '\n\nAUC Score: {}\n\n'.format(roc_auc_score(y_test, pred), 3))


# In[57]:


import pickle

with open('model', 'wb') as f:
    pickle.dump(best_xgb, f)


# In[58]:


# with open('model', 'rb') as f:
#     mp = pickle.load(f)
    
    
# mp.score(X_test, y_test)
