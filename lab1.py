import numpy as np
import pandas as pd

# Function to load data from an Excel file and prepare it for training
def dataload(filename):
    # Read data from Excel file into a pandas DataFrame
    df = pd.read_excel(filename, index_col=0)
    # Convert DataFrame to numpy array
    data = df.to_numpy()
    # Extract features (X) and labels (y) from the data
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y

# Function to split the dataset into training and testing sets
def train_test_split(x, y):
    # Set seed for reproducibility
    np.random.seed(42)
    # Shuffle indices
    shuffle = np.random.permutation(len(x))
    # Shuffle features and labels accordingly
    xrandom = x[shuffle]
    yrandom = y[shuffle]
    # Split data into training and testing sets (70% training, 30% testing)
    xtrain = xrandom[:int(0.7*len(x))]
    ytrain = yrandom[:int(0.7*len(y))]
    xtest  = xrandom[int(0.7*len(x)):]
    ytest  = yrandom[int(0.7*len(y)):]
    return xtrain, ytrain, xtest, ytest

# Function to train the KNN model and normalize the data
def train(X, y):
    # Calculate minimum and maximum values for each feature
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    # Calculate ranges for each feature
    ranges = maxs - mins
    # Normalize the data
    normalized_data = (X - mins) / ranges
    # Return trained model as a dictionary containing normalized data, minimums, ranges, and labels
    return {"NormalizedData": normalized_data,
            "Minimums": mins,
            "Ranges": ranges,
            "Labels": y}

# Function to test the KNN model
def test(model, xtest, kValues):
    # Extract normalized data from the model
    X = model["NormalizedData"]
    # Normalize the testing data using the model's minimums and ranges
    xtest_n = (xtest - model["Minimums"]) / model["Ranges"]
    # Calculate distances between testing data and training data
    distances = np.sqrt(np.sum((X[:, np.newaxis] - xtest_n[np.newaxis, :]) ** 2, axis=2).astype(float))
    # Combine distances with labels
    distancesW_labels = np.hstack((distances, model["Labels"][:, np.newaxis]))
    # Predict labels for testing data based on different values of k
    predictedLabels = []
    for k in kValues:
        # Sort distancesW_labels based on distances
        sorted_distancesW_labels = distancesW_labels[distancesW_labels[:, 0].argsort()]
        # Select the k nearest neighbors
        KNN = sorted_distancesW_labels[:k]
        # Find the most common label among the nearest neighbors
        unique, counts = np.unique(KNN[:, 1], return_counts=True)
        predictedLabel = unique[np.argmax(counts)]
        predictedLabels.append(predictedLabel)
    return predictedLabels

# Function to calculate precision, recall, and false positive rate
def metrics(HumanFinalAns, AIFinalAns):
    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = np.sum(np.logical_and(AIFinalAns == 1, AIFinalAns == HumanFinalAns))
    FP = np.sum(np.logical_and(AIFinalAns == 1, AIFinalAns != HumanFinalAns))
    TN = np.sum(np.logical_and(AIFinalAns == 0, AIFinalAns == HumanFinalAns))
    FN = np.sum(np.logical_and(AIFinalAns == 0, AIFinalAns != HumanFinalAns))
    # Calculate precision, recall, and false positive rate
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    falsePositiveRate = FP / (FP + TN + 1e-8)
    return precision, recall, falsePositiveRate

# Load data from Excel file
X, y = dataload("data.xlsx")

# Split data into training and testing sets
xtrain, ytrain, xtest, ytest = train_test_split(X, y)

# Train the KNN model
model = train(xtrain, ytrain)

# Define values of k for testing
kValues = [1, 3, 5]

# Test the model for each value of k and calculate performance metrics
predictedLabels = test(model, xtest, kValues)
for i, k in enumerate(kValues):
    precision, recall, falsePositiveRate = metrics(ytest, predictedLabels[i])
    print(f"\nFor k = {k}:")
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("False Positive Rate: {:.3f}".format(falsePositiveRate))

# Find the best k based on accuracy
accuracies = [np.mean(predictedLabels[i] == ytest) for i in range(len(kValues))]
bestK = kValues[np.argmax(accuracies)]
print("\nBest k: ", bestK)
