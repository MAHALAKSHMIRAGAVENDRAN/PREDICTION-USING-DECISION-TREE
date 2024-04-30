#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries/Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
dataset = pd.read_csv('Iris.csv')
dataset


# In[3]:


dataset.shape


# In[4]:


dataset.columns


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset['Species'].value_counts()


# In[9]:


#Pie plot to show the overall types of Iris classifications
dataset['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.09,0.09,0.09])


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))
sns.heatmap(dataset.corr(), cmap='CMRmap', annot=True, linewidths=2)
plt.title("Correlation Graph", size=20)
plt.show()


# In[11]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = dataset.loc[:, features].values   #defining the feature matrix
y = dataset.Species


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

#Defining the decision tree classifier and fitting the training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[13]:


DecisionTreeClassifier()


# In[14]:


#Visualizing the decision tree
from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= dataset.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# In[15]:


y_pred = dtree.predict(X_test)
y_pred


# In[17]:


from sklearn import metrics
     
#Checking the accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[19]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[21]:


dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# In[22]:


dtree.predict([[5, 3.6, 1.4 , 0.2]])


# In[30]:


dtree.predict([[90, 35, 500, 19]])

