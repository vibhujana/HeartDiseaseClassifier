import numpy as np
import pandas
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

file_name = "heart_2020_cleaned.csv"
col_names = ["HeartDisease","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]
df = pd.read_csv(file_name)
df['Smoking'] = df['Smoking'].map({'Yes':1 ,'No':0})#Smoking
df['AlcoholDrinking'] = df['AlcoholDrinking'].map({'Yes':1 ,'No':0})#AlcoholDrinking
df['Stroke'] = df['Stroke'].map({'Yes':1 ,'No':0})#Stroke
df['DiffWalking'] = df['DiffWalking'].map({'Yes':1 ,'No':0})#DiffWalking
df['Sex'] = df['Sex'].map({'Male':1 ,'Female':0})#Sex(Male,Female)
df['AgeCategory'] = df['AgeCategory'].map({'40-44':40,'45-49':45,'50-54':50,'55-59':55,'60-64':60,'65-69':65,'70-74':70,'75-79':75,"80 or older":80, "18-24":18,"25-29":25,"30-34":30,"35-39":35})#AgeCategory(55-59, 80 or older, 40 - 44, 75 - 79,...)
df['Race'] = df['Race'].map({'White':0 ,'Black':1,'Asian':2,"Hispanic":3, "Other":5, "American Indian/Alaskan Native":4})#Race(White, Black, Asian, Hispanic)
df['Diabetic'] = df['Diabetic'].map({'Yes':1 ,'No':0,"No, borderline diabetes":2, "Yes (during pregnancy)": 3})#Diabetic
df['PhysicalActivity'] = df['PhysicalActivity'].map({'Yes':1 ,'No':0})#PhysicalActrivty(Very Good...)
df['GenHealth'] = df['GenHealth'].map({'Poor':0,"Fair":1,"Good":2,"Very good":3,"Excellent":4})#GenHealth(Very Good, Good, Excellent, Fair, Poor)
df['Asthma'] = df['Asthma'].map({'Yes':1 ,'No':0})#Asthma
df['KidneyDisease'] = df['KidneyDisease'].map({'Yes':1 ,'No':0})#KidneyDisease
df['SkinCancer'] = df['SkinCancer'].map({'Yes':1 ,'No':0})#SkinCare
df['HeartDisease'] = df['HeartDisease'].map({'Yes':1 ,'No':0})
df = df.astype(int)
x = df[col_names[1:]]
y = df.HeartDisease
y = y.to_frame()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
# print(y_test)
dtc = DecisionTreeClassifier()
# print((x_train))
# print((y_train).to_string())
dtc = dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
# dot_data = StringIO()
# export_graphviz(dtc, out_file=dot_data, rounded= True, filled = True, special_characters=True, feature_names=col_names[1:], class_names= ['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('heartdiseaseentire.png')
# Image(graph.create_png())






#TODO: study the optimized model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = col_names[1:],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('heartdisease.png')
Image(graph.create_png())


def predictPerson(clf):
    col_names = ["HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth", "MentalHealth",
                 "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime",
                 "Asthma", "KidneyDisease", "SkinCancer"]
    data = np.array([18,0,0,0,15,30,0,1,18,0,0,1,2,9,0,0,0])
    print(data)
    data = pd.DataFrame([data], columns= col_names[1:])
    prediction = clf.predict(data)
    return prediction
print(predictPerson(clf))