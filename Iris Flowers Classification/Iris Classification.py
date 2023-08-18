# DataFlair Iris Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # As per the iris dataset information

# Load the data
df = pd.read_csv('C:\\Users\\VINAY BHARADWAJ\\Desktop\\internship\\ml\\Iris Flowers Classification\\iris.data', names=columns)
df.head()
sns.pairplot(df, hue='Class_labels')

# Seperate features and target  
data = df.values
a1 = data[:,0:4]
a2 = data[:,4]

# Calculate avarage of each features for all classes
a2_Data = np.array([np.average(a1[:, i][a2==j].astype('float32')) for i in range (a1.shape[1]) for j in (np.unique(a2))])
a2_Data_reshaped = a2_Data.reshape(4, 3)
a2_Data_reshaped = np.swapaxes(a2_Data_reshaped, 0, 1)
a1_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the avarage
plt.bar(a1_axis, a2_Data_reshaped[0], width, label = 'Setosa')
plt.bar(a1_axis+width, a2_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(a1_axis+width*2, a2_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(a1_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
a1_train, a1_test, a2_train, a2_test = train_test_split(a1, a2, test_size=0.2)

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(a1_train, a2_train)

# Predict from the test dataset
predictions = svn.predict(a1_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(a2_test, predictions)

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(a2_test, predictions))

a1_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(a1_new)
print("Prediction of Species: {}".format(prediction))

# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(a1_new)
