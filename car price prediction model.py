#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


car_data=pd.read_csv('D:\car dataset.csv')


# In[3]:


car_data.head()


# In[4]:


# Checking the number of rows and column
car_data.shape


# In[5]:


# info about the dataset
car_data.info()


# In[6]:


#checking the null values
car_data.isnull().sum()


# In[7]:


# Checking the distribution of catageorical dat
print(car_data.fuel.value_counts())
print(car_data.seller_type.value_counts())
print(car_data.transmission.value_counts())
print(car_data.owner.value_counts())


# In[8]:


# encoding " fuel , seller_type and transmission"

car_data.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3}},inplace=True)
car_data.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True)
car_data.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)
car_data.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}},inplace=True)


# In[9]:


car_data.head()


# In[10]:


# splitting the data target
x=car_data.drop(['name','selling_price'],axis=1)
y=car_data['selling_price']


# In[11]:


print(x)


# In[12]:


print(y)


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.1,random_state=2)


# # model training
# 1 Random Forest

# In[14]:


#create random forest regressor
rf_regressor=RandomForestRegressor(n_estimators=100,random_state=2)


# In[15]:


#train model 
rf_regressor.fit(x_train,y_train)


# In[16]:


#make prediction on test case
y_test_pred=rf_regressor.predict(x_test)


# In[17]:


#make prediction on train set
y_train_pred=rf_regressor.predict(x_train)


# In[18]:


#evaluate the model 
mse_test=mean_squared_error(y_test,y_test_pred)
mse_train=mean_squared_error(y_train,y_train_pred)

print('Mean Squared Error (Test):',mse_test)
print('Mean Squared Error (Train):',mse_train)


# In[21]:


plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[22]:


plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# # 2 Gradient boosting

# In[30]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# In[24]:


# create and train the Gradient Boosting model 
model = GradientBoostingRegressor()
model.fit(x_train, y_train)


# In[27]:


# make prediction on the test set
y_test_pred = model.predict(x_test)


# In[28]:


# make prediction on the train set
y_train_pred = model.predict(x_train)


# In[33]:


#Calculate the R-squared score for test set
r2_test = r2_score(y_test, y_test_pred)
print('R-squared:', r2_test)


# In[35]:


#Calculate the R-squared score for train set
r2_train = r2_score(y_train, y_train_pred)
print('R-squared:', r2_train)


# In[36]:


plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[37]:


plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[ ]:




