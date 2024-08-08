import rosbag
import pandas as pd
import os
import rospy_message_converter.message_converter as converter

CURRENT_FILE = os.getcwd()
file_paths = [
    # "out_2024-03-04-19-08-30_one_person_moving",
    # "out_2024-03-04-19-09-45_standing",
    # "out_2024-03-04-19-10-28_two_people_moving",
    # "out_2024-03-04-19-11-38_one_person_standing_one_moving",
    # "out_2024-03-04-19-13-21"
    "out_2024-03-25-10-31-23_one_person_moving",
    "out_2024-03-25-10-32-56_one_standing_one_moving",
    "out_2024-03-25-10-34-45_two_people_moving",
    "out_2024-03-25-10-37-34_two_in_parallel"
]

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
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)

def process_data(data):
    # Identifying if any key contains a list of dictionaries
    list_keys = [key for key, value in data.items() if (len(value) >0 and isinstance(value, list)) and all(isinstance(item, dict) for item in value)]
    print(list_keys, data)
    if len(list_keys) == 0:
        return [flatten_dict(data)] # Return original data if no suitable list is found

    res = []
    for key in list_keys:
        print(key)
        for item in data[key]:
            print(item)
            new_row = data.copy()
            new_row.update({key: item})
            res.append(flatten_dict(new_row))

    return res

def save_data(results_path, data_path):
    bag = rosbag.Bag(data_path + '.bag')
    topics = ['/objectposes', '/person_state_estimation/person_states', '/qualisys/pedestrian/pose', '/qualisys/pedestrian/velocity', '/qualisys/tinman/pose', '/qualisys/tinman/velocity']
    create_dir(results_path)
    datasets = {}

    for topic, msg, t in bag.read_messages(topics=topics):
        d = converter.convert_ros_message_to_dictionary(msg)
        flattened_data = process_data(d)  # Process data to handle lists

        if topic not in datasets:
            datasets[topic] = pd.DataFrame(flattened_data)
        else:
            datasets[topic] = pd.concat([datasets[topic], pd.DataFrame(flattened_data)], ignore_index=True)

    for topic, df in datasets.items():
        data_file_path = os.path.join(results_path, topic.replace('/', '_') + '.csv')
        df.to_csv(data_file_path, index=False)

if __name__ == "__main__":
    for file_name in file_paths:
        results_path = os.path.join(CURRENT_FILE, 'results', file_name)
        data_path = os.path.join(CURRENT_FILE, 'data', file_name)
        save_data(results_path, data_path)