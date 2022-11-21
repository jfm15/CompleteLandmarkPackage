import lib

from sklearn.metrics import confusion_matrix


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

    tn, fp, fn, tp = confusion_matrix(ground_truth_diagnoses, predicted_diagnoses).ravel()
    accuracy = 100 * float(tn + tp) / float(n)
    precision = 100 * float(tp) / float(tp + fp)
    recall = 100 * float(tp) / float(tp + fn)
    return n, tn, fp, fn, tp, precision, recall, accuracy
