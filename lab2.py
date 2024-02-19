import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Data Preparation and Processing
digits = datasets.load_digits()
data = digits["data"]
target = digits["target"]
x_train, x_test, y_train, y_test = \
    train_test_split(data, target, test_size=0.3, shuffle=False)

# Training kNN and Hyperparameter Optimization
HyperParameterKinKNN = [3, 5, 7, 11]
kNNAccuracies = np.zeros_like(HyperParameterKinKNN).astype(float)

for index, k in enumerate(HyperParameterKinKNN):
    clf_knn = KNeighborsClassifier(k)
    clf_knn.fit(x_train, y_train)
    AIFinalAns = clf_knn.predict(x_test)
    Probabilities  = clf_knn.predict_proba(x_test)
    kNNAccuracies[index] = accuracy_score(y_test, AIFinalAns)
print("Best k is ", HyperParameterKinKNN[kNNAccuracies.argmax()])
BestK = HyperParameterKinKNN[kNNAccuracies.argmax()]

# ------------------------------------------------
# Task 1: Training SVM and Hyperparameter Optimization
clf_svm = SVC(kernel='rbf')
HP_Gamma = {"gamma": [0.0001, 0.001, 0.01, 0.1]}
grid_search = GridSearchCV(clf_svm, HP_Gamma, cv=6)
grid_search.fit(x_train, y_train)
best_gamma = grid_search.best_params_['gamma']

HyperParaGinSVM = [0.0001, 0.001, 0.01, 0.1]
SVMAccuracies = np.zeros_like(HyperParaGinSVM).astype(float)
for index, G in enumerate(HyperParaGinSVM):
    clf_svm = SVC(kernel='rbf', gamma=G)
    clf_svm.fit(x_train, y_train)
    AISVMFinalAns = clf_svm.predict(x_test)
    SVMAccuracies[index] = accuracy_score(y_test, AISVMFinalAns)

KNNCM = metrics.confusion_matrix(y_test, AIFinalAns)
KNNCR = metrics.classification_report(y_test, AIFinalAns)
SVMCM = metrics.confusion_matrix(y_test, AISVMFinalAns)
SVMCR = metrics.classification_report(y_test, AISVMFinalAns)

print("KNN Classification Report: \n", KNNCR)
print("SVM Classification Report: \n", SVMCR)

print("KNN Confusion Matrix: \n", KNNCM)
print("SVM Confusion Matrix: \n", SVMCM)

def showMeTheErrors(x_test, AIFinalAns, y_test, classifier):
    ToughImages = x_test[AIFinalAns - y_test != 0]
    ToughTarget = AIFinalAns[AIFinalAns - y_test != 0]
    ToughTarget = ToughTarget[0:5]
    RealTarget  = y_test[AIFinalAns - y_test != 0]
    for i in range(len(ToughTarget)):
        plt.imshow(ToughImages[i].reshape(8, 8),
                   cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(classifier)
        plt.suptitle(f"Prediction: {ToughTarget[i]} - Ground Truth: {RealTarget[i]}")
        plt.show()

y_test_binarized = label_binarize(y_test, classes=np.unique(target))

knn_probs = clf_knn.predict_proba(x_test)

svm_probs = clf_svm.predict_proba(x_test)

n_classes = len(np.unique(target))
knn_fpr = dict()
knn_tpr = dict()
knn_auc = dict()
svm_fpr = dict()
svm_tpr = dict()
svm_auc = dict()

for i in range(n_classes):
    knn_fpr[i], knn_tpr[i], _ = roc_curve(y_test_binarized[:, i], knn_probs[:, i])
    knn_auc[i] = auc(knn_fpr[i], knn_tpr[i])

    svm_fpr[i], svm_tpr[i], _ = roc_curve(y_test_binarized[:, i], svm_probs[:, i])
    svm_auc[i] = auc(svm_fpr[i], svm_tpr[i])
knn_fpr["micro"], knn_tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), knn_probs.ravel())
knn_auc["micro"] = auc(knn_fpr["micro"], knn_tpr["micro"])

svm_fpr["micro"], svm_tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), svm_probs.ravel())
svm_auc["micro"] = auc(svm_fpr["micro"], svm_tpr["micro"])
plt.figure(figsize=(8, 6))

plt.plot(knn_fpr["micro"], knn_tpr["micro"], color='blue', lw=2, label=f'KNN (AUC = {knn_auc["micro"]:.2f})')
plt.plot(svm_fpr["micro"], svm_tpr["micro"], color='red', lw=2, label=f'SVM (AUC = {svm_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title("SVM is the better classifier in this case", color="red")
plt.suptitle('ROC Curve - KNN vs SVM')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
