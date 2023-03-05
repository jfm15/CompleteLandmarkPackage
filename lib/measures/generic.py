import lib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

def measure(predicted_points, target_points, suffix, measure_name):

    function_name = ".".join(["lib", "measures", suffix, measure_name])
    predicted_angle = eval(function_name)(predicted_points)
    target_angle = eval(function_name)(target_points)
    dif = abs(target_angle - predicted_angle)
    return predicted_angle, target_angle, dif


def diagnose_individual(predicted_points, target_points, suffix, diagnosis_name):
    function_name = ".".join(["lib", "measures", suffix, diagnosis_name])
    predicted_diagnosis = eval(function_name)(predicted_points)
    true_diagnosis = eval(function_name)(target_points)
    return predicted_diagnosis, true_diagnosis


def diagnose_set(aggregated_scaled_points, dataset_target_scaled_points, suffix, diagnosis_name):

    n = len(aggregated_scaled_points)
    predicted_diagnoses = []
    ground_truth_diagnoses = []
    for i in range(n):
        predicted_points = aggregated_scaled_points[i]
        target_points = dataset_target_scaled_points[i]
        predicted_diagnosis, true_diagnosis = diagnose_individual(predicted_points, target_points, suffix, diagnosis_name)
        # use extend because some diagnosis contain left and right
        predicted_diagnoses.extend(predicted_diagnosis)
        ground_truth_diagnoses.extend(true_diagnosis)

    #check if predicted diagnosis has multiple classes
    if np.unique(np.array(predicted_diagnoses)).size < 2:
        #only one class for classification problem
        tn, fp, fn, tp = confusion_matrix(ground_truth_diagnoses, predicted_diagnoses).ravel()
        total = tn + fp + fn + tp
        if total == 0:
            accuracy = 0
        else:
            accuracy = 100 * float(tn + tp) / float(total)

        if tp+fp == 0:#
            precision = 0
        else:
            precision = 100 * float(tp) / float(tp + fp)
        if tp+fn == 0:
            recall = 0
            pass
        else:
            recall = 100 * float(tp) / float(tp + fn)
    else:
        #multi class so outputs will be an array for tn, fp, fn, tp
        confusion_matrix_multiclasses = multilabel_confusion_matrix(ground_truth_diagnoses, predicted_diagnoses)
        #find values for all
        accuracy = np.array([])
        precision = np.array([])
        recall =  np.array([])
        total = np.array([])
        tn, fp, fn, tp = np.array([]),np.array([]),np.array([]),np.array([])
        for _class in confusion_matrix_multiclasses:
            _tn, _fp, _fn, _tp = _class.ravel()
            _total = _tn + _fp + _fn + _tp
            if _total == 0:
                _accuracy = 0
            else:
                _accuracy = 100 * float(_tn + _tp) / float(_total)

            if _tp+_fp == 0:#
                _precision = 0
            else:
                _precision = 100 * float(_tp) / float(_tp + _fp)
            if _tp+_fn == 0:
                _recall = 0
                pass
            else:
                _recall = 100 * float(_tp) / float(_tp + _fn)
            
            precision = np.append(precision, _precision)
            accuracy = np.append(accuracy, _accuracy)
            recall = np.append(recall, _recall)
            total =  np.append(total, _total)
            
            tn, fp, fn, tp = np.append(tn,_tn),np.append(fp,_fp), np.append(fn,_fn),np.append(tn,_tp)


    return total, tn, fp, fn, tp, precision, recall, accuracy
