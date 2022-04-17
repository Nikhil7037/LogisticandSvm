#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('stroke.csv')
print(df)

#checking for null values
print(df.isnull().sum(axis=0))

#drop NaN values
df = df.dropna()


# In[3]:


#filtering data and removing data with Unknow value types in dataframe
df = df[df['smoking_status'] != 'Unknown']
x = df['smoking_status'].value_counts()
x = x.reset_index()
print(x)


# In[4]:


#filtering data and removing data with other in gender value types in dataframe
df = df[df['gender'] != 'Other']
x = df['gender'].value_counts()
x = x.reset_index()
print(x)





# In[5]:


#converting gender as 0 and 1
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
df['gender'] = label.fit_transform(df['gender'])


str_data = df.select_dtypes(include=['object'])
str_dt = df.select_dtypes(include=['object'])
int_data = df.select_dtypes(include = ['integer','float'])
int_dt = df.select_dtypes(include = ['integer','float'])
label = LabelEncoder()
df = str_data.apply(label.fit_transform)
df= df.join(int_data)



# In[6]:


#lets build machine learning model
X = df[['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi']]
y = df['stroke']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

x_val = X_train.values
y_val = y_train.values




# In[7]:


# Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size=10, epochs=100)
classifier.fit(x_val, y_val, batch_size=1000, epochs=10)




# In[8]:


# first example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[101, 1, 1, 202.21,1,2,1,2,1,36.6]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi'])
fruit_prediction = classifier.predict(testFruit)
print(fruit_prediction)


# In[9]:


print('Performance:')
print(classifier.evaluate(X_test, y_test))


# In[12]:


y_pred=classifier.predict(X_test)
print(y_pred)


# In[11]:





# In[ ]:




