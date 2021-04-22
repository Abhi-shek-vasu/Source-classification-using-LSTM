import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Loading the data
dataFrameA = pd.read_excel('data/train&test/A.xlsx', header=None)
dataFrameG = pd.read_excel('data/train&test/G.xlsx', header=None)

#Loading Testfiles
dataFramePred1 = pd.read_excel('data/pred/Testfile_1.xlsx', header=None)
dataFramePred2 = pd.read_excel('data/pred/Testfile_2.xlsx', header=None)
dataFramePred3 = pd.read_excel('data/pred/Testfile_3.xlsx', header=None)

#Dropping first 6 columns
x_A = np.array(dataFrameA.drop([0,1,2,3,4,5], axis=1))
x_G = np.array(dataFrameG.drop([0,1,2,3,4,5], axis=1))
x_pred1 = np.array(dataFramePred1.drop([0,1,2,3,4,5], axis=1))
x_pred2 = np.array(dataFramePred2.drop([0,1,2,3,4,5], axis=1))
x_pred3 = np.array(dataFramePred3.drop([0,1,2,3,4,5], axis=1))

#vertical stacking
x = np.vstack((x_A, x_G))

#Reshaping
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

x_pred1 = np.reshape(x_pred1, (x_pred1.shape[0], 1, x_pred1.shape[1]))  
x_pred2 = np.reshape(x_pred2, (x_pred2.shape[0], 1, x_pred2.shape[1]))
x_pred3 = np.reshape(x_pred3, (x_pred3.shape[0], 1, x_pred3.shape[1]))

#Asigning Labels
y = np.hstack((np.zeros(x_A.shape[0]), np.ones(x_G.shape[0])))

#Data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Model creation

model = Sequential()
    
# Add Model
model.add(LSTM(32,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))

#Compile Model
model.add(Dense(2,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
     optimizer=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5),
     metrics=['accuracy'])

model.summary()

#Fit model
model.fit(x_train,y_train,epochs=10,validation_split=0.2)

# evaluate model

model.evaluate(x_test, y_test, verbose= 1)

#confusion Matrix
y_act=y_test
y_predcm=model.predict(x_test)
y_predcm=np.argmax(y_predcm.reshape((y_predcm.shape[0],y_predcm.shape[2])), axis=1)
con_mat= confusion_matrix(y_act, y_predcm).ravel()
tn,fp,fn,tp=con_mat
display=ConfusionMatrixDisplay(confusion_matrix=con_mat.reshape(2,2))
display.plot()
tpr=tp/(tp + fn)
tnr=tp/(tn + fp)
fdr=tp/(fp + tp)
npv=tn/(tn + fn)

#model save and load
model.save("lstm.h5")

model = tf.keras.models.load_model("lstm.h5")

#predect model for Testfile_1
y_pred1 = model.predict(x_pred1)
y_pred1 = np.reshape(y_pred1, (y_pred1.shape[0], y_pred1.shape[2]))
y_pred1 = np.argmax(y_pred1, axis= 1)
print('Result of Testfile_1=y_pred1')
print(y_pred1)


#predect model for Testfile_2
y_pred2 = model.predict(x_pred2)
y_pred2 = np.reshape(y_pred2, (y_pred2.shape[0], y_pred2.shape[2]))
y_pred2 = np.argmax(y_pred2, axis= 1)
print('Result of Testfile_2=y_pred2')
print(y_pred2)

#predect model for Testfile_3
y_pred3 = model.predict(x_pred3)
y_pred3 = np.reshape(y_pred3, (y_pred3.shape[0], y_pred3.shape[2]))
y_pred3 = np.argmax(y_pred3, axis= 1)
print('Result of Testfile_3=y_pred3')
print(y_pred3)