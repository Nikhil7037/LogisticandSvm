
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('stroke.csv')
#print(df)

#checking for null values
print(df.isnull().sum(axis=0))

#drop NaN values
df = df.dropna()



#filtering data and removing data with Unknow value types in dataframe
df = df[df['smoking_status'] != 'Unknown']
x = df['smoking_status'].value_counts()
x = x.reset_index()
print(x)

#filtering data and removing data with other in gender value types in dataframe
df = df[df['gender'] != 'Other']
x = df['gender'].value_counts()
x = x.reset_index()
print(x)



#divinding into groups with stroke and nostroke as 0 and 1
print(df.groupby('stroke').size())

#glucose level based on stroke
df_avg_glucose_level= df.groupby(['stroke'],as_index=False).avg_glucose_level.mean()
print("Average glucose level based on stroke and no stroke")
print(df_avg_glucose_level)

#Median glucose level based on stroke
df_median_glucose_level= df.groupby(['stroke'],as_index=False).avg_glucose_level.median()
print("Median glucose level based on stroke and no stroke")
print(df_median_glucose_level)

#standard deviation based on stroke:
df_sd_glucose_level= df.groupby(['stroke'],as_index=False).avg_glucose_level.describe()
print("sd glucose level based on stroke and no stroke")
print(df_sd_glucose_level)

#stroke count for males and females
df_test=df[df['stroke']==0].groupby(['gender','stroke']).size().reset_index(name='count')
print(df_test)

df_test2=df[df['stroke']==1].groupby(['gender','stroke']).size().reset_index(name='count')
print(df_test2)

#stroke for married and nit married
df_test=df[df['stroke']==0].groupby(['ever_married','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['ever_married','stroke']).size().reset_index(name='count')
print(df_test)

#stroke for heart_diseases
df_test=df[df['stroke']==0].groupby(['heart_disease','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['heart_disease','stroke']).size().reset_index(name='count')
print(df_test)

#stroke for hypertension
df_test=df[df['stroke']==0].groupby(['hypertension','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['hypertension','stroke']).size().reset_index(name='count')
print(df_test)

print('-----------------------')
#stroke for work_type
df_test=df[df['stroke']==0].groupby(['work_type','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['work_type','stroke']).size().reset_index(name='count')
print(df_test)


#bmi based on stroke
df_avg_bmi_level= df.groupby(['stroke'],as_index=False).bmi.mean()
print("Average bmi based on stroke and no stroke")
print(df_avg_bmi_level)

#Median bmi based on stroke
df_median_bmi_level= df.groupby(['stroke'],as_index=False).bmi.median()
print("bmi median based on stroke and no stroke")
print(df_median_bmi_level)

#standard deviation based on stroke:
df_sd_bmi_level= df.groupby(['stroke'],as_index=False).bmi.describe()
print("sd bmi based on stroke and no stroke")
print(df_sd_bmi_level)

#age based on stroke
df_avg_age= df.groupby(['stroke'],as_index=False).age.mean()
print("Average age based on stroke and no stroke")
print(df_avg_age)

#Median bmi based on stroke
df_median_age= df.groupby(['stroke'],as_index=False).age.median()
print("age median based on stroke and no stroke")
print(df_median_age)

#standard deviation based on stroke:
df_sd_age= df.groupby(['stroke'],as_index=False).age.describe()
print("sd bmi based on stroke and no stroke")
print(df_sd_age)

#stroke based on residency
df_test=df[df['stroke']==0].groupby(['Residence_type','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['Residence_type','stroke']).size().reset_index(name='count')
print(df_test)

#converting gender as 0 and 1
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
df['gender'] = label.fit_transform(df['gender'])
print(df)

str_data = df.select_dtypes(include=['object'])
str_dt = df.select_dtypes(include=['object'])
int_data = df.select_dtypes(include = ['integer','float'])
int_dt = df.select_dtypes(include = ['integer','float'])
label = LabelEncoder()
df = str_data.apply(label.fit_transform)
df= df.join(int_data)
print(df.head())

#lets build machine learning model
X = df[['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi']]
y = df['stroke']




from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

from imblearn.under_sampling import RandomUnderSampler
oversample = RandomOverSampler(sampling_strategy=0.5)

# fit the object to the training data.
X_train ,y_train = oversample.fit_resample(X,y)



# kernals could be: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’. Try them.
from sklearn import svm
clfsvm = svm.SVC(kernel='linear')
clfsvm.fit(X_train, y_train)
y_pred=clfsvm.predict(X_test)

#### See which data points are critical #####
# get the support vectors
print("clfsvm support vectors: {}".format(clfsvm.support_vectors_))
# get indices of support vectors
print("clfsvm support vector indices: {}".format(clfsvm.support_))
# get number of support vectors for each class
print("clfsvm # of support vectors in each class: {}".format(clfsvm.n_support_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(clfsvm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clfsvm.score(X_test, y_test)))

fruit_prediction = clfsvm.predict(X_test)
compare = pd.DataFrame({'true': y_test, 'predicted': fruit_prediction})
print("true vs predicted\n", compare)

# first example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[101, 1, 1, 202.21,1,2,1,2,1,36.6]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi'])
fruit_prediction = clfsvm.predict(testFruit)
print(fruit_prediction)

# second example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[75, 1, 1, 212.21,1,2,1,2,1,55]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi'])
fruit_prediction = clfsvm.predict(testFruit)
print(fruit_prediction)

# second example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[35, 1, 1, 212.21,1,2,1,2,1,55]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi'])
fruit_prediction = clfsvm.predict(testFruit)
print(fruit_prediction)

# second example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[25, 0, 0, 120.00,1,2,1,2,1,39]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level','gender','work_type','ever_married','Residence_type','smoking_status','bmi'])
fruit_prediction = clfsvm.predict(testFruit)
print(fruit_prediction)









from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)

from sklearn.metrics import plot_confusion_matrix

#plot_confusion_matrix(clfsvm , X_test, y_test)
#plt.show()
