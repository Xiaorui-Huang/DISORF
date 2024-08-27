import sys
import rospy
import cv2
import numpy as np
import struct
import json
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from packet_subscriber import PacketDeserializer
import tf.transformations as tf_trans
import time
import argparse

def packet_subscriber(port):
    """This function receives data from packet and then
    publish the data to ros"""
    packet_deserializer = PacketDeserializer(port)
    packet_deserializer.start()

    rospy.init_node('image_and_pose_publisher', anonymous=True) # shuts down on SIGINT (Ctrl+C) by default
    image_pub = rospy.Publisher("/image_raw", Image, queue_size=10000)
    pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=10000)

    dataframe_received = 0

    while not rospy.is_shutdown():
        img = None
        value = None
        json_parsed = None

        # read with timeout to check for shutdown
        try:
            packet = packet_deserializer.read()
        except Exception as e:
            continue


        if(dataframe_received == 0):
            start_time = time.time()
        dataframe_received += 1
        if(dataframe_received != 0):
            time_elapsed = time.time() - start_time
            print("###########################################")
            print(f"Dataframe received: {dataframe_received}, Time elapsed: {time_elapsed}, Dataframe rate: {dataframe_received / time_elapsed}")
            print("###########################################")

        if packet['type'] == 1:  # If the packet is an image
            # size = struct.unpack('!I', packet['payload'][:4])[0]  # get the size of the image data
            array = np.frombuffer(packet['payload'][8:], dtype=np.uint8)  # get the image data
            img = cv2.imdecode(array, cv2.IMREAD_COLOR)
            print("Got an image")
            if img is None:
                print("Failed to decode image.")
            elif img.size == 0:
                print("Decoded image is empty.")

        packet = packet_deserializer.read()
        if packet['type'] == 2:  # If the packet is a double
            value = struct.unpack('!d', packet['payload'])[0]
            # print(f"Received value: {value}")

        packet = packet_deserializer.read()
        if packet['type'] == 3:  # If the packet is a JSON
            json_str = packet['payload'].decode()
            json_parsed = json.loads(json_str)
            # print(f"Received json: {json_parsed}")

        if img is None or value is None or json_parsed is None:
            print("broken dataframe")
        else:
            print("Good dataframe")

            # publish the dataframe through ROS
            # OpenCV
            height, width, channels = img.shape
            is_color = channels == 3
            
            # Create Image message
            img_msg = Image()
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = "bgr8" if is_color else "mono8"
            img_msg.is_bigendian = False
            img_msg.step = channels * width  # Full row length in bytes
            img_msg.data = img.tostring()

            # Create PoseStamped message
            pose_msg = PoseStamped()
            print(json_parsed["transform_matrix"])

            transform_mat = np.array(json_parsed["transform_matrix"])
            pose_msg.pose.position.x = transform_mat[0, 3]
            pose_msg.pose.position.y = transform_mat[1, 3]
            pose_msg.pose.position.z = transform_mat[2, 3]
                        
            # Get the rotation part of the matrix
            rotation_mat = transform_mat[:3, :3]

            # Convert rotation matrix to quaternion
            quaternion = tf_trans.quaternion_from_matrix(transform_mat)

            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]

            if not rospy.is_shutdown():
                time_now = rospy.Time.now()
                img_msg.header.stamp = time_now
                pose_msg.header.stamp = time_now

                image_pub.publish(img_msg)
                pose_pub.publish(pose_msg)

def main():
    parser = argparse.ArgumentParser(description='Program to listen for local end keyframes.')
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5001,
        help='Port to listen on (default: 5001)'
    )
    
    args = parser.parse_args()

    port = args.port

    print(f"Listening on {port} for keyframes...")
    packet_subscriber(port)

if __name__ == "__main__":
    main()