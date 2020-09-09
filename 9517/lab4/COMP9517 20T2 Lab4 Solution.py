import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

test_size = 0.25
n_neighbors = 3


digits = load_digits()

# plt.imshow(np.reshape(digits.data[0], (8, 8)), cmap='gray')
# plt.title('Label: %i\n' % digits.target[0], fontsize=25)
# plt.show()

print("Image Data and Labels Shape:", digits.data.shape,  digits.target.shape)
print("Image Data Size:", digits.data[0].size)

# Display the first 10 images and labels
plt.figure(figsize=(20, 10))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap='gray')
    plt.title('Label: %i\n' % label, fontsize=25)
plt.show()

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target,
    test_size=test_size, random_state=0)

# Initialize the KNN model
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(x_train, y_train)
# Evaluate/predict the test data
knn_predictions = model.predict(x_test)

# Initialize the SGD model
model = SGDClassifier(max_iter=5)
model.fit(x_train, y_train)
# Evaluate/predict the test data
sgd_predictions = model.predict(x_test)

# Initialize the DT model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
# Evaluate/predict the test data
dt_predictions = model.predict(x_test)

# Display the results
print("COMP9517 Week 5 Lab - Solution")
print("\nTest size = %.2f" % test_size)

print("KNN Accuracy:  %0.3f \t Recall: %0.3f " % (metrics.accuracy_score(y_test, knn_predictions), metrics.recall_score(y_test, knn_predictions, average='macro')))
print("SGD Accuracy:  %0.3f \t Recall: %0.3f " % (metrics.accuracy_score(y_test, sgd_predictions), metrics.recall_score(y_test, sgd_predictions, average='macro')))
print("DT Accuracy:   %0.3f \t Recall: %0.3f " % (metrics.accuracy_score(y_test, dt_predictions), metrics.recall_score(y_test, dt_predictions, average='macro')))

print("\nKNN Confusion Matrix:")
print(metrics.confusion_matrix(y_test, knn_predictions))
