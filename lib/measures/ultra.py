from lib.utils import get_angle


def alpha_angle(points):

    baseline = points[1, :] - points[0, :]
    bony_roof_line = points[3, :] - points[2, :]

    return get_angle(baseline, bony_roof_line)


def beta_angle(points):

    baseline = points[1, :] - points[0, :]
    cartilage_roof_lines = points[4, :] - points[2, :]

    return get_angle(baseline, cartilage_roof_lines)


def ddh(points):

    aa = alpha_angle(points)
    ba = beta_angle(points)

    ##return zero or one for classification 
    
    # if aa < 50:
    #     return [1]
    # else:
    #     return [0]

    grf_dic = {
    "1": {'a':'>60', 'b':'NA', 'd': 'Normal: Discharge Patient'},
    "2a": {'a':'50-59', 'b':'NA', 'd': 'Normal: Clinical Review -/+ treat'},
    "2b": {'a':'50-59', 'b':'NA', 'd': 'Abnormal: Clinical Review -/+ treat'},
    "2c": {'a':'43-49', 'b':'<77', 'd':'Abnormal: Clinical Review + treat'},
    "D": {'a':'43-49', 'b':'>77', 'd': 'Abnormal: Clinical Review + treat'}, 
    "3": {'a':'<43', 'b':'Unable to calculate', 'd': 'Abnormal: Clinical Review + treat'},
    "4": {'a':'<43', 'b':'Unable to calculate', 'd': 'Abnormal: Clinical Review + treat'},
    }

    # based on above the classes will be 
    # 0: 'Normal: Discharge Patient'
    # 1: 'Normal: Clinical Review -/+ treat'
    # 2: 'Abnormal: Clinical Review -/+ treat'
    # 3: 'Abnormal: Clinical Review + treat'

    #currently based only off of alpha angle for classification.
    if aa >= 60:
        return [0]
    elif aa > 50 and aa < 60:
        return [1]
    elif aa > 43 and aa < 50:
        return [2]
    elif aa < 43:
        return [3]
    else:
        raise ValueError
