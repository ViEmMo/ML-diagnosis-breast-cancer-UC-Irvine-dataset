# Import necessary libraries
import pandas as pd

# Scikit-learn for data preprocessing and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Keras for building and training neural network models
from keras.models import Sequential
from keras.layers import Dense

# Load the breast cancer dataset from the UCI Machine Learning Repository
breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                           names=["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"])

# Display the first few rows of the dataset
breast_cancer.head()

# Extract features (X) and labels (Y) from the dataset
X = breast_cancer.drop(['diagnosis', 'id'], axis=1).values
Y = breast_cancer['diagnosis'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Encode the categorical labels (Malignant/Benign) to numerical values
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Standardize the feature values for better model performance
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# Build a sequential neural network model using Keras
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and stochastic gradient descent optimizer
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the neural network on the training data for 100 epochs
model.fit(X_train, y_train, epochs=100)