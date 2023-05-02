
def alpha_angles(ax, points, color):
    _, right_center, right_neck_point, _, left_center, left_neck_point, right_cam_point, left_cam_point = points

    # left
    ax.plot([left_center[0], left_cam_point[0]], [left_center[1], left_cam_point[1]], color=color, linewidth=0.5)
    ax.plot([left_center[0], left_neck_point[0]], [left_center[1], left_neck_point[1]], color=color, linewidth=0.5)

    # right
    ax.plot([right_center[0], right_cam_point[0]], [right_center[1], right_cam_point[1]], color=color, linewidth=0.5)
    ax.plot([right_center[0], right_neck_point[0]], [right_center[1], right_neck_point[1]], color=color, linewidth=0.5)


def compare_alpha_angles(ax, predicted_points, target_points):

    alpha_angles(ax, target_points, "lime")
    alpha_angles(ax, predicted_points, "red")

    '''
    ax.scatter([right_center[0], right_cam_point[0], right_neck_point[0], left_center[0], left_cam_point[0], left_neck_point[0]],
               [right_center[1], right_cam_point[1], right_neck_point[1], left_center[1], left_cam_point[1], left_neck_point[1]],
               color='green', s=20)
    '''

    #for i, positions in enumerate(target_points):
    #    idx = important_indices[i]
    #    ax.text(positions[0], positions[1], "{}".format(idx + 1), color="yellow", fontsize="small")


def lce_angles(ax, points, color):

    right_lat_point, right_center, _, left_lat_point, left_center, _, _, _ = points

    # left
    ax.plot([left_center[0], left_lat_point[0]], [left_center[1], left_lat_point[1]], color=color, linewidth=0.5)
    ax.plot([left_center[0], left_center[0]], [left_center[1], left_center[1] - 50], color=color, linewidth=0.5)

    # right
    ax.plot([right_center[0], right_lat_point[0]], [right_center[1], right_lat_point[1]], color=color, linewidth=0.5)
    ax.plot([right_center[0], right_center[0]], [right_center[1], right_center[1] - 50], color=color, linewidth=0.5)


def compare_lce_angles(ax, predicted_points, target_points):

    lce_angles(ax, target_points, "lime")
    lce_angles(ax, predicted_points, "red")

    '''
    ax.scatter([right_center[0], right_lat_point[0], left_center[0], left_lat_point[0]],
               [right_center[1], right_lat_point[1], left_center[1], left_lat_point[1]], color='green', s=20)
    '''