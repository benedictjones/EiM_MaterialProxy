import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mod_settings.GenParam import GenSimParam, LoadPrm
from mod_load.FetchDataObj import FetchDataObj

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC as svc
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load in param template
tprm = LoadPrm(param_file='')

# Gen final prm file
prm = GenSimParam(param_file=tprm)  # Produce PAramater File


for dataset in ['c2DDS', 'MMDS']:  # ['d2DDS', 'c2DDS', 'BankNote', 'MMDS']

    print(" ### %s ###" % (dataset))
    prm['DE']['training_data'] = dataset

    # Make load object
    lobj = FetchDataObj(prm)

    trainX, trainY = lobj.fetch_data('train')
    testX, testY = lobj.fetch_data('test')

    #

    #

    print("\n** Random Forest Algorithm **")
    rfc_object = rfc(n_estimators=100, random_state=0)
    rfc_object.fit(trainX, trainY)
    predicted_labels = rfc_object.predict(testX)
    #print(classification_report(testY, predicted_labels))
    #print(confusion_matrix(testY, predicted_labels))
    print("accuracy:", accuracy_score(testY, predicted_labels), ", error:", 1-accuracy_score(testY, predicted_labels))
    """
    fig = plt.figure()
    Y = rfc_object.predict_proba(testX)
    plt.scatter(testX[:,0], testX[:,1])
    plt.title('Random Forest Algorithm (probability)')
    plt.ylabel('a2')
    plt.xlabel('a1')
    plt.colorbar()
    #"""

    #

    #

    print("\n** Support Vector Machine **")
    trainY2d = trainY.reshape(-1, 1)
    testY2d = testY.reshape(-1, 1)

    svc_object = svc(kernel='linear')
    svc_object.fit(trainX, trainY2d)
    predicted_labels = svc_object.predict(testX)
    #print(classification_report(testY2d, predicted_labels))
    #print(confusion_matrix(testY2d, predicted_labels))
    print("SVM linear:", accuracy_score(testY2d, predicted_labels), ", error:", 1-accuracy_score(testY2d, predicted_labels))

    svc_object = svc(kernel='poly')
    svc_object.fit(trainX, trainY2d)
    predicted_labels = svc_object.predict(testX)
    #print(classification_report(testY2d, predicted_labels))
    #print(confusion_matrix(testY2d, predicted_labels))
    print("SVM poly:", accuracy_score(testY2d, predicted_labels), ", error:", 1-accuracy_score(testY2d, predicted_labels))
    """
    fig = plt.figure()
    Y = svc_object.decision_function(testX)
    plt.scatter(testX[:,0], testX[:,1], c=Y)
    plt.title('SVM')
    plt.ylabel('a2')
    plt.xlabel('a1')
    plt.colorbar()
    #"""

    # .reshape(-1, 1)

    #

    print("\n** Ridge Regression **")
    model = RidgeCV(normalize=False)  # alphas=(0.01, 0.1, 1.0, 10.0),
    model.fit(trainX, trainY)
    Y = model.predict(testX)

    class_out = []
    for val in Y:
        if val >= 1.5:
            class_out.append(2)
        else:
            class_out.append(1)
    #print(classification_report(testY2d, class_out))
    #print(confusion_matrix(testY2d, class_out))
    print("accuracy:", accuracy_score(testY, class_out), ", error:", 1-accuracy_score(testY, class_out))
    """
    fig = plt.figure()
    plt.scatter(testX[:,0], testX[:,1], c=Y)
    plt.title('Ridge Regression')
    plt.ylabel('a2')
    plt.xlabel('a1')
    plt.colorbar()
    #"""

    #

    #

    print("\n** Logistic Regression **")
    model = LogisticRegression()
    model.fit(trainX, trainY)
    predicted_labels = model.predict(testX)
    #print(classification_report(testY, predicted_labels))
    #print(confusion_matrix(testY, predicted_labels))
    print("accuracy:", accuracy_score(testY, predicted_labels), ", error:", 1-accuracy_score(testY, predicted_labels))
    """
    fig = plt.figure()
    Y = model.decision_function(testX)
    plt.scatter(testX[:,0], testX[:,1], c=Y)
    plt.title('Logistic Regression')
    plt.ylabel('a2')
    plt.xlabel('a1')
    plt.colorbar()
    #"""

    # ################################

    #

    #

    #

    plt.show()

#

#

#

#

# fin
