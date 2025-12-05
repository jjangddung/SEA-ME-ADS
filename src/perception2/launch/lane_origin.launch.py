from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    
    lane_mask = Node(
        package='perception',
        executable='lane_mask_node.py',
        name='lane_mask',
        # output='screen'
    )

    obj_detect = Node(
        package='perception',
        executable='yolo_detect.py',
        name='obj_detect',
        # output='screen'
    )

    pc_projector = Node(
        package='perception',
        executable='pc_projector',
        name='pc_projector',
        # output='screen'
    )

    
    global_path = Node(
        package='perception',
        executable='global_path.py',
        name='global_path',
        # output='screen'
    )

    map_making = Node(
        package='perception',
        executable='map_making.py',
        name='map_making',
        # output='screen'
    )

    pf_localizer = Node(
        package='perception',
        executable='lane_pf_localizer',
        name='pf_localizer',
        # output='screen'
    )

    return LaunchDescription([
        lane_mask,
        obj_detect,
        pc_projector,
        global_path,
        map_making,
        pf_localizer
    ])
