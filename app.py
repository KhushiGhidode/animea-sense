#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:\\Users\\lab-12\\khushiGhidode\\flask\\anemia.csv")
df.info()
df.shape
df.isnull().sum()
results = df['Result'].value_counts()
results.plot(kind = 'bar', color=['blue', 'green'] )
plt.xlabel('Result')
plt.ylabel('frequency')
plt.title('count of Result')
plt.show()
from sklearn.utils import resample
majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]
major_downsample = resample(majorclass, replace=False, n_samples=len(minorclass) ,random_state=42)
df=pd.concat([major_downsample, minorclass])
print(df['Result'].value_counts())
results_balanced = df['Result'].value_counts()
results_balanced.plot(kind = 'bar', color =['blue', 'green'] )
plt.xlabel('Result')
plt.ylabel('frequency')
plt.title('count of Result (Balanced)')
plt.show()
df.describe()
output = df['Gender'].value_counts()
output.plot(kind = 'bar', color=['orange', 'green'])
plt.xlabel('Gender')
plt.ylabel('frequency')
plt.title('Gender count')
plt.show()
sns.displot(df['Hemoglobin'], kde =True)
plt.figure(figsize=(6,6))
ax= sns.barplot(y= df['Hemoglobin'],x = df['Gender'],hue = df['Result'],ci =None)
ax.set(xlabel=['male','female'])
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Mean Hemoglobin by Gender and Result")
plt.show()
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
Y = df['Result']
Y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y ,test_size=0.2, random_state=20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
acc_lr = accuracy_score(y_test,y_pred)
c_lr = classification_report(y_test,y_pred)
print('Accuracy Score: ',acc_lr)
print(c_lr)
from sklearn.ensamble import RandomForestClassifier
random_forest= RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc_rf = accuracy_score(y_test,y_pred)
c_rf = classification_report(y_test,y_pred)
print('Accuracy Score: ',acc_rf)
print(c_rf)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(x_test)
acc_nb = accuracy_score(y_test,y_pred)
c_nb = classification_report(y_test,y_pred)
print('Accuracy Score: ',acc_nb)
print(c_nb)
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_test)
acc_gbc = accuracy_score(y_test,y_pred)
c_gbc = classification_report(y_test,y_pred)
print('Accuracy Score: ',acc_gbc)
print(c_gbc)
prediction = GBC.predict([[0,11.6,22.3,30.9,74.5]])
prediction[0]
if prediction[0] == 0:
    print ("you don't have any Anemia Disease")
elif prediction[0] ==1:
     print ("you have Anemia Disease")
model = pd.DataFrame({'Model':['linear regression','Decision tree Classifier', 'RandomForest classifier',
                               'Gaussian navie Bayes','Support vector Classifier','Gradient Boost classifier'],
                      'Score':[acc_lr,acc_dt,acc_rf,acc_nb,acc_svc,acc_gbc],
                     })
model
import pickle
import warnings
pickle.dump(GBC,open("model.pkl", "wb"))
import numpy as np
import pickle
import pandas as pd
from flask import flask, request, render_template
app = flask(__name__, static_url_path='/flask/static')
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
     return render_template('Index.html')
@app.route('/predict', methods=["post"])
def predict(): 
   Gender = float(request.form["Gender"])
   Hemoglobin = float(request.form["Hemoglobin"])
   MCH = float(request.form["MCH"])
   MCHC = float(request.form["MCHC"])
   MCV = float(request.form["MCV"])
   features_values = np.array([['Gender','Hemoglobin', 'MCH','MCHC','MCV']]) 
   print(df)
   prediction = model.predict(df)
   print(prediction[0])
   result = prediction[0]
   if prediction[0] == 0:
    print ("you don't have any Anemia Disease")
   elif prediction[0] ==1:
     print ("you have  Anemia Disease")
   text = "hence, based on calculation:" 
   return render_template("predict.html", prediction_text_test + str(result))
if __name__== "_main_":
     app.run(debug= True, port = 5000)
