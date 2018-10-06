
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target

features.head()


# In[3]:


#RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=0)
rf.fit(features, targets)


# In[4]:


target_predicted = rf.predict(features)
target_predicted[0]


# In[9]:


import matplotlib.pyplot as plt
#visualizing difference between predicted and original data with a horizontal line at 0
# Plots on line 0 means value predicted is matching with original data
get_ipython().magic('matplotlib inline')
plt.scatter(target_predicted,target_predicted- targets,c="b",s=40,alpha=0.9, label = 'Train data')
#plt.scatter(pred_test,pred_test- Y_test,c="r",s=40,alpha=0.9,  label = 'Test data')
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')

# Y label
plt.ylabel("Residuals")

## plot title
plt.title("Residual plot using entire data")

plt.show()


# In[7]:


# Spliting data for training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features,targets,test_size=0.3, random_state=5)

rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=0)
rf.fit(X_train, Y_train)


# In[10]:


pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

plt.scatter(pred_train,pred_train- Y_train,c="b",s=40,alpha=0.9, label = 'Train data')
plt.scatter(pred_test,pred_test- Y_test,c="r",s=40,alpha=0.9,  label = 'Test data')
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')

# Y label
plt.ylabel("Residuals")

## plot title
plt.title("Residual plot using training and test data")

plt.show()

