import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report


##Evaluates the performance of a binary classifier against labelled data.
# @param data - a dataframe containing 3 columns: class (binary values), predicted(binary values) and 
#probability (the predicition the model has made for each value, prior to converting to a binary classification, number between 0 and 1).
# @returns - Confusion matrix plot, ROC curve plot, and a small report.

def model_evaluator(data):
    
    #partition processed data into vectors
    actualClass = data.class
    predictedClass = data.predicted
    probability = data.probabilty 
    
    #build a confusion matrix
    cm = confusion_matrix(actualClass, predictedClass)
    
    
    TruePositive = cm[1, 1]
    TrueNegative = cm[0,0]
    FalsePositive = cm[0,1]
    FalseNegative = cm[1,0]
    
    numberOfPositives = TruePositive + FalseNegative
    numberOfNegatives = TrueNegative + FalsePositive
    
    #calculate the Null accuracy
    null_accuracy = 1 - actualClass.mean()
    
    #define the model accuracy
    model_accuracy = metrics.accuracy_score(actualClass, predictedClass)
    
    #Generate a metrics report
    report = metrics.classification_report(actualClass, predictedClass, output_dict = True)
    
    #calculate the model performance over the null accuracy
    performance_over_null = model_accuracy - null_accuracy
    
    #Calculate the Specificity of the model 
    specificity = TrueNegative / (TrueNegative + FalsePositive)
    
    #Calculate the True positive rate, false positive rate, and thresholds to plot a rock curve
    fpr, tpr, thresholds = metrics.roc_curve(actualClass, probability)
    
    #Calculate the Area under the ROC Curve
    rocAuc = metrics.roc_auc_score(actualClass, probability)
    
    #generate figure
    fig = plt.figure(figsize=(12, 20))
    spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.5, figure=fig)
    
    #plot confusion matrix in pos 0,0
    confusionMatrixLabels = ['Normal Traffic', 'Intrusion']
    confusionMatrixColourMap = plt.cm.Blues
    confusionMatrix = fig.add_subplot(spec[0, 0])
    confusionMatrix.set_aspect('equal')
    confusionMatrix.imshow(cm, interpolation='nearest', cmap = confusionMatrixColourMap)
    confusionMatrix.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), 
                         xticklabels = CM_labels, yticklabels = CM_labels,
                         ylabel='True class', xlabel='Predicted class')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            confusionMatrix.text(j, i, (format(cm[i, j])), ha="center", va="center", color="red", size = 'larger')
    
    cmLabels = ['TN', 'FP', 'FN', 'TP' ]
    a = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            confusionMatrix.text(j + 0.3, i + 0.4, (cmLabels[a]), ha ='center', va="baseline", color = 'red', size = 'larger')
            if a < 4:
                a += 1
    a=0          
    for i in range(cm.shape[a]):
        if a == 0:
            confusionMatrix.text(j+0.8, i, ('Total:\n %d' % (numberOfNegatives)), ha ='center', va="center", color = 'black', size = 'larger')
            a += 1
        else:
            confusionMatrix.text(j+0.8, i, ('Total:\n %d' % (numberOfPositives)), ha ='center', va="center", color = 'black', size = 'larger')
    
    #plot roc curve in position 0,1 
    rocCurve = fig.add_subplot(spec[0, 1])
    rocCurve.set_aspect('equal')
    rocCurve.plot(fpr, tpr, color='red', lw=2, label = 'ROC area = %0.2f)' % roc_auc )
    rocCurve.set(xlim = [0.0, 1.0], ylim = [0.0, 1.0], xlabel = 'True Positive Rate (Sensitivity)', ylabel = 'True Positive Rate (Sensitivity)' )
    rocCurve.legend(loc="lower right")
    
    #print report
    print('The performance of this model over the null accuracy is %2.2f%%\nModel Sensitivity: %2.2f%% \nModel Specificity: %2.2f%% \nModel F1 Score: %2.2f' 
      % ((performance_over_null *100), (report['weighted avg']['recall'] *100), (specificity * 100), (report['weighted avg']['f1-score'])))
