import lib


def measure(predicted_points, target_points, suffix, measure_name):

    function_name = ".".join(["lib", "measures", suffix, measure_name])
    predicted_angle = eval(function_name)(predicted_points)
    target_angle = eval(function_name)(target_points)
    dif = abs(target_angle - predicted_angle)
    return predicted_angle, target_angle, dif