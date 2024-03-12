import rosbag
import pandas as pd

CURRENT_FILE = './src/svea_vision/svea_vision/'
file_paths = ["out_2024-03-04-19-08-30_one_person_moving", "out_2024-03-04-19-09-45_standing", 
              "out_2024-03-04-19-10-28_two_people_moving", "out_2024-03-04-19-11-38_one_person_standing_one_moving", 
              "out_2024-03-04-19-13-21"]

for path in file_paths:
    # The bag file should be in the same directory as your terminal
    bag = rosbag.Bag(CURRENT_FILE + path + '.bag')
    topics = ['/objectposes', '/person_state_estimation/person_states', '/qualisys/pedestrian/pose', '/qualisys/pedestrian/velocity', '/qualisys/tinman/pose', '/qualisys/tinman/velocity']
    column_names = ['topic', 'seq', 'stamp_secs', 'stamp_nsecs',
        'positionx', 'positiony', 'positionz', 
        'orientationx', 'orientationy', 'orientationz', 'orientationw']
    df = pd.DataFrame(columns=column_names)

    for topic, msg, t in bag.read_messages(topics=topics):
        print(msg)
        print(topic)
        print('\n')
        seq = msg.header.seq
        stamp_secs = msg.header.stamp.secs
        stamp_nsecs = msg.header.stamp.nsecs

        if '/pose' in topic:
            positionx = msg.pose.position.x
            positiony = msg.pose.position.y
            positionz = msg.pose.position.z
            orientationx = msg.pose.orientation.x
            orientationy = msg.pose.orientation.y
            orientationz = msg.pose.orientation.z
            orientationw = msg.pose.orientation.w
        elif '/velocity' in topic:
            positionx = msg.twist.linear.x
            positiony = msg.twist.linear.y
            positionz = msg.twist.linear.z
            orientationx = msg.twist.angular.x
            orientationy = msg.twist.angular.y
            orientationz = msg.twist.angular.z
            orientationw = '-'
        elif '/objectposes' in topic:
            positionx = msg.objects[0].pose.pose.position.x
            positiony = msg.objects[0].pose.pose.position.y
            positionz = msg.objects[0].pose.pose.position.z
            orientationx = msg.objects[0].pose.pose.orientation.x
            orientationy = msg.objects[0].pose.pose.orientation.y
            orientationz = msg.objects[0].pose.pose.orientation.z
            orientationw = msg.objects[0].pose.pose.orientation.w
        elif '/person_states' in topics:
            if len(msg.personstate) == 0:
                positionx = None
                positiony = None
                positionz = None
                orientationx = None
                orientationy = None
                orientationz = None
                orientationw = None
            else:
                positionx = msg.personstate.position.x
                positiony = msg.personstate.position.y
                positionz = msg.personstate.position.z
                orientationx = msg.personstate.orientation.x
                orientationy = msg.personstate.orientation.y
                orientationz = msg.personstate.orientation.z
                orientationw = msg.personstate.orientation.w


        dataset = pd.DataFrame({
            'topic': [topic], 
            'seq': [seq], 
            'stamp_secs': [stamp_secs], 
            'stamp_nsecs': [stamp_nsecs],
            'positionx': [positionx],
            'positiony': [positiony],
            'positionz': [positionz],
            'orientationx': [orientationx],
            'orientationy': [orientationy],
            'orientationz': [orientationz],
            'orientationw': [orientationw]})
        
        df = pd.concat([df, dataset])

    df.to_csv(CURRENT_FILE + path + '.csv')