#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[2]:


train_data.head(5)


# In[3]:


test_data.head(5)


# In[4]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    return ms


# In[5]:


missingdata(train_data)


# In[6]:


missingdata(test_data)


# In[7]:


train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace= True)


# In[8]:


test_data['Fare'].fillna(test_data['Fare'].median(), inplace= True)


# In[9]:


train_data['Age'].fillna(train_data['Age'].mode()[0], inplace= True)
test_data['Age'].fillna(test_data['Age'].mode()[0], inplace= True)


# In[10]:


train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1


# In[11]:


train_data.isnull().sum()


# In[12]:


test_data.isnull().sum()


# In[13]:


all_data = [train_data, test_data]


# In[14]:


for dataset in all_data:
    dataset["Age"] = dataset["Age"].astype("int64")
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])


# In[15]:


traindata=train_data
testdata=test_data


# In[16]:


alldata = [traindata, testdata]


# In[17]:


for dataset in alldata:
    drop_column = ['Fare', 'Cabin','Name', 'Ticket','PassengerId']
    dataset.drop(drop_column, axis=1, inplace = True)


# In[18]:


traindata = pd.get_dummies(traindata, columns = ["Sex","Embarked","Fare_bin"],
                             prefix=["Sex","Em_type","Fare_type"])


# In[19]:


testdata = pd.get_dummies(testdata, columns = ["Sex","Embarked","Fare_bin"],
                             prefix=["Sex","Em_type","Fare_type"])


# In[20]:


sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix 


all_features = traindata.drop("Survived",axis=1)
Targeted_feature = traindata["Survived"]
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[22]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)

model.fit(X_train,y_train)
prediction_rm=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is',round(accuracy_score(prediction_rm,y_test)*100,2))

kfold = KFold(n_splits=10, random_state=22, shuffle = True) 
result_rm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)


# In[23]:


final_result = model.predict(testdata)


# In[24]:


# output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': final_result})
# output.to_csv('submission.csv', index=False)
# import pickle
# filename = 'titan_model.pkl'
# pickle. dump(model, open(filename, 'wb'))


# In[27]:


testdata.head(5)


# In[29]:


testdata.shape


# In[ ]:




