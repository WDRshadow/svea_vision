import rosbag
import pandas as pd
import os
import rospy_message_converter.message_converter as converter

CURRENT_FILE = os.getcwd()
file_paths = ["out_2024-03-04-19-08-30_one_person_moving", "out_2024-03-04-19-09-45_standing", 
              "out_2024-03-04-19-10-28_two_people_moving", "out_2024-03-04-19-11-38_one_person_standing_one_moving", 
              "out_2024-03-04-19-13-21"]


def create_dir(dir):
    """
    Input:
        dir: a directory to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
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
            children = flatten_dict(v, new_key, sep=sep).items()
            if children:  # If the dictionary has children, extend items without adding the parent dictionary
                items.extend(children)
            else:  # If the dictionary is empty, add it as a value
                items.append((new_key, v))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:  # For lists of non-dictionary items, add each item
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def save_data(results_path, data_path):

    # The bag file should be in the same directory as your terminal
    bag = rosbag.Bag(data_path + '.bag')
    topics = ['/objectposes', '/person_state_estimation/person_states', '/qualisys/pedestrian/pose', '/qualisys/pedestrian/velocity', '/qualisys/tinman/pose', '/qualisys/tinman/velocity']
    create_dir(results_path)
    datasets = {}

    for topic, msg, t in bag.read_messages(topics=topics):
        d = converter.convert_ros_message_to_dictionary(msg)
        print(d)
        print(topic)
        data = flatten_dict(d)
        print(data)

        print('\n')

        if topic in datasets:
            datasets[topic] = pd.concat([datasets[topic], pd.DataFrame([data], columns=data.keys())])    
        else:
            datasets[topic] = pd.DataFrame([data], columns=data.keys())
                                           
    for topic in topics:
        data_file_path = results_path + topic.replace('/', '_') + '.csv'    
        datasets[topic].to_csv(data_file_path, columns=datasets[topic].keys(), index=False)


if __name__ == "__main__":
    for file_name in file_paths:
        results_path = CURRENT_FILE + '/results/' + file_name + '/'
        data_path = CURRENT_FILE + '/data/' + file_name
        save_data(results_path, data_path)
