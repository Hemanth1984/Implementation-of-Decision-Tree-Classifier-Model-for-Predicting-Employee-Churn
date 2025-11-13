# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: E Hemachandran
RegisterNumber:212224230093

import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
df["left"].value_counts()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
print("NAME : Bharath K")
print("REG NO : 212224230036")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
conf=confusion_matrix(y_test,y_pred)
conf
cla=classification_report(y_test,y_pred)
cla
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
Head:

<img width="1097" height="180" alt="511973759-f8ffed60-873b-404e-be82-3508dd1cee77" src="https://github.com/user-attachments/assets/8fe7d8b9-86ba-464b-a471-f84d27658111" />


Info:

<img width="487" height="335" alt="511973913-aff32a05-4d48-4278-b98b-3b9075ecd03b" src="https://github.com/user-attachments/assets/995f00e9-e3c4-4e8d-9a28-bd9148cf1b76" />

Null:

<img width="253" height="215" alt="511974111-5eab1174-a23c-41ce-8228-32f0300aa04d" src="https://github.com/user-attachments/assets/03de509e-9ad7-412d-99e3-84becd2260ce" />

head:

<img width="1093" height="180" alt="511974263-3c6984a8-25e5-45bc-a341-4e9968001b9d" src="https://github.com/user-attachments/assets/4d1ad7a7-f55e-423d-89ba-de1b14936e52" />

value_counts:

<img width="230" height="77" alt="511974385-19aa8367-5fae-41c4-a66d-3314aeb34675" src="https://github.com/user-attachments/assets/445d3e86-820d-492b-b81c-87b2bdc966cf" />

x.head():

<img width="1047" height="187" alt="511974566-950d8da8-8099-46bf-9350-f780fb71fe1d" src="https://github.com/user-attachments/assets/f3120224-4d7a-48dc-9f2f-f1ccbecdafa7" />


dt.fit(x_train,y_train):

![WhatsApp Image 2025-11-13 at 09 20 00_059b8ae3](https://github.com/user-attachments/assets/5c95bda7-e791-4f0b-9553-1abb4b7367c8)

y_pred:

<img width="397" height="40" alt="511975328-66fd8b64-046f-418c-81a7-e00f68a6886c" src="https://github.com/user-attachments/assets/3b9719a7-10eb-4ca5-b16c-a3662e713a98" />

acc:

<img width="77" height="32" alt="511975459-2c33eccf-70bb-4d0c-9cf2-e57e1feb0df3" src="https://github.com/user-attachments/assets/1c6878eb-c4ce-46a5-a5e8-fc73dacbda3e" />

conf:

<img width="327" height="55" alt="511975612-bb0d49b6-6504-455c-9670-06385ee95e2a" src="https://github.com/user-attachments/assets/dac1fbac-06e0-4a02-a607-29abe237aea5" />

classification:
<img width="1118" height="70" alt="511975796-a87aa5e9-c830-4127-a3da-20ae842b49c4" src="https://github.com/user-attachments/assets/a9ef5dbc-f42a-4156-aaf6-23d7f9f17494" />

predict:

<img width="222" height="31" alt="511976094-1597cde6-c689-4a3e-a720-7f5ebd510cc7" src="https://github.com/user-attachments/assets/7e3503d5-cec1-49ce-93b8-fc1438647e06" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
