import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import numpy as np


def callback(data):
    rospy.loginfo("Receiving pointcloud")
    pc_array = ros_numpy.numpify(data)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    np.save('../ros/pointcloud/temp.npy', pc)


def pointcloud_saver():
    rospy.init_node('point_cloud_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, callback)
    rospy.spin()


if __name__ == '__main__':
    pointcloud_saver()
