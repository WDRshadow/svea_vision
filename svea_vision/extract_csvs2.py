import rosbag
import pandas as pd
import os
import rospy_message_converter.message_converter as converter

CURRENT_FILE = './src/svea_vision/svea_vision/'
file_paths = ["out_2024-03-04-19-08-30_one_person_moving", "out_2024-03-04-19-09-45_standing", 
              "out_2024-03-04-19-10-28_two_people_moving", "out_2024-03-04-19-11-38_one_person_standing_one_moving", 
              "out_2024-03-04-19-13-21"]


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


# Function to flatten the dictionary
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_data(path):
    create_dirs([path])

    # The bag file should be in the same directory as your terminal
    bag = rosbag.Bag(path + '.bag')
    topics = ['/objectposes', '/person_state_estimation/person_states', '/qualisys/pedestrian/pose', '/qualisys/pedestrian/velocity', '/qualisys/tinman/pose', '/qualisys/tinman/velocity']

    for topic, msg, t in bag.read_messages(topics=topics):
        d = converter.convert_ros_message_to_dictionary(msg)
        print(d)
        print(topic)
        print('\n')

        dataset = pd.DataFrame([flatten_dict(d)])        
        dataset.to_csv(path + topic + '.csv', columns=dataset.keys())


if __name__ == "__main__":
    for path in file_paths:
        save_data(CURRENT_FILE + path)
