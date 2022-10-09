

def alpha_angles(ax, predicted_points, target_points):

    right_center = target_points[1]
    right_cam_point = target_points[6]
    right_neck_point = target_points[2]

    left_center = target_points[4]
    left_cam_point = target_points[7]
    left_neck_point = target_points[5]

    # left
    ax.plot([left_center[0], left_cam_point[0]], [left_center[1], left_cam_point[1]], color='lime')
    ax.plot([left_center[0], left_neck_point[0]], [left_center[1], left_neck_point[1]], color='lime')

    # right
    ax.plot([right_center[0], right_cam_point[0]], [right_center[1], right_cam_point[1]], color='lime')
    ax.plot([right_center[0], right_neck_point[0]], [right_center[1], right_neck_point[1]], color='lime')

    ax.scatter([right_center[0], right_cam_point[0], right_neck_point[0], left_center[0], left_cam_point[0], left_neck_point[0]],
               [right_center[1], right_cam_point[1], right_neck_point[1], left_center[1], left_cam_point[1], left_neck_point[1]],
               color='green', s=20)


    #for i, positions in enumerate(target_points):
    #    idx = important_indices[i]
    #    ax.text(positions[0], positions[1], "{}".format(idx + 1), color="yellow", fontsize="small")


def lce_angles(ax, predicted_points, target_points):

    right_center = target_points[1]
    right_lat_point = target_points[0]

    left_center = target_points[4]
    left_lat_point = target_points[3]

    # left
    ax.plot([left_center[0], left_lat_point[0]], [left_center[1], left_lat_point[1]], color='lime')
    ax.plot([left_center[0], left_center[0]], [left_center[1], left_center[1] - 50], color='lime')

    # right
    ax.plot([right_center[0], right_lat_point[0]], [right_center[1], right_lat_point[1]], color='lime')
    ax.plot([right_center[0], right_center[0]], [right_center[1], right_center[1] - 50], color='lime')

    ax.scatter([right_center[0], right_lat_point[0], left_center[0], left_lat_point[0]],
               [right_center[1], right_lat_point[1], left_center[1], left_lat_point[1]], color='green', s=20)