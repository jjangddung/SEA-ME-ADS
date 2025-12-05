from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    
    # lane_detection = Node(
    #     package='perception',
    #     executable='lane_detection.py',
    #     name='lane_detection',
    #     # output='screen'
    # )

    lane_detection = Node(
        package='perception',
        executable='lane_detection',
        name='lane_detection',
        # output='screen'
    )

    #planning으로 뺄 것들

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

    region = Node(
        package='perception',
        executable='checking_region.py',
        name='region',
        # output='screen'
    )

    return LaunchDescription([
        lane_detection,
        global_path,
        map_making,
        pf_localizer,
        region
    ])
