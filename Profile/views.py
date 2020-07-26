from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
global model_svm


global model

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def User(request):
    if request.method == 'GET':
       return render(request, 'User.html', {})


def Admin(request):
    if request.method == 'GET':
       return render(request, 'Admin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'AdminScreen.html', context)
      else:
       context= {'data':'Invalid Login'}
       return render(request, 'Admin.html', context)


def importdata(): 
    balance_data = pd.read_csv('C:/Users/AISHWARYA THAKUR/Desktop/ai/Profile/dataset/dataset.txt')
    balance_data = balance_data.abs()
    rows = balance_data.shape[0]  # gives number of row count
    cols = balance_data.shape[1]  # gives number of col count
    return balance_data 

def splitdataset(balance_data):
    X = balance_data.values[:, 0:8] 
    y_= balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    print(Y)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    return train_x, test_x, train_y, test_y
#experiment_SVM
def splitdataset2(balance_data):
    X = balance_data.values[:, 0:8] 
    Y= balance_data.values[:, 8]
    print(Y)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3,random_state=109)
    return train_X, test_X, train_Y, test_Y

def UserCheck(request):
    global model
    global model_svm  
    if request.method == 'POST':
        data = request.POST.get('t1', False)
        input = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n';        
        input+=data+"\n"
        a=[]
        b=data.split(',')
        for i in b:
            a.append(int(i))     
        f = open("C:/Users/AISHWARYA THAKUR/Desktop/ai/Profile/dataset/test.txt", "w")
        f.write(input)
        f.close()
        test = pd.read_csv('C:/Users/AISHWARYA THAKUR/Desktop/ai/Profile/dataset/test.txt')
        test = test.values[:, 0:8] 
        predict = model.predict_classes(test)
        msg = ''
        if len(a) != 8:
            msg='Enter correct number of parameters'
        else:    
            if str(predict[0]) == '0':
                msg = "Account is genuine"
            if str(predict[0]) == '1':
                msg = "Account is fake"    
    context= {'data':msg}       
      
    return render(request, 'User.html', context)

def GenerateModel(request):
    global model
    global model_svm
    a=''
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(2, activation='softmax', name='output'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Artificial Neural Network Model Summary: ')
    print(model.summary())
    model.fit(train_x, train_y, verbose=2, batch_size=10, epochs=100)
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100
   
    #svm
    
    train_X,test_X,train_Y,test_Y=splitdataset2(data)
    model_svm = SVC(probability=True)
    model_svm_pred = model_svm.fit(train_X,train_Y)
    svmpred = model_svm_pred.predict(test_X)
    svm_acc = accuracy_score(test_Y,svmpred)*100
   
    context= {'data':'ANN Accuracy : '+str(ann_acc),'data1':'SVM Accuracy : '+str(svm_acc)}
    
    return render(request, 'AdminScreen.html', context)

def ViewTrain(request):
    if request.method == 'GET':
       strdata = '<table border=1 align=center width=100%><tr><th><font size=4 color=black>Account Age</th><th><font size=4 color=black>Gender</th><th><font size=4 color=black>User Age</th><th><font size=4 color=black>Link Description</th> <th><font size=4 color=black>Status Count</th><th><font size=4 color=black>Friend Count</th><th><font size=4 color=black>Location</th><th><font size=4 color=black>Location IP</th><th><font size=4 color=black>Profile Status</th></tr><tr>'
       data = pd.read_csv('C:/Users/AISHWARYA THAKUR/Desktop/ai/Profile/dataset/dataset.txt')
       rows = data.shape[0]  # gives number of row count
       cols = data.shape[1]  # gives number of col count
       for i in range(rows):
          for j in range(cols):
             strdata+='<td><font size=3 color=black>'+str(data.iloc[i,j])+'</font></td>'
          strdata+='</tr><tr>'
       context= {'data':strdata}
       return render(request, 'ViewData.html', context)