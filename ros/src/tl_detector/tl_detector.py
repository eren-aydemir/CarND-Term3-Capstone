#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import scipy
from scipy import spatial
import numpy as np
import time

#STATE_COUNT_THRESHOLD = 5
STATE_COUNT_THRESHOLD = 2 # for check the classification accuracy

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        self.kd_tree_for_tl = None
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.kd_tree_traffilight = None
        self.last_light_index = -1
        self.current_state = None

        rospy.logerr("Classifier for {}".format(self.config['type']))
        self.light_classifier = TLClassifier(self.config['type'])


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        waypoints_size = len(self.waypoints.waypoints)
        if not self.kd_tree_for_tl:
            pts = np.ndarray([waypoints_size,2])
            for i in range(waypoints_size):
                x=waypoints.waypoints[i].pose.pose.position.x
                y=waypoints.waypoints[i].pose.pose.position.y
                pts[i][0]=x
                pts[i][1]=y
            self.kd_tree_for_tl = scipy.spatial.KDTree(pts)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #DONE implement
        if not self.kd_tree_for_tl:
            return 0
        x = pose.position.x
        y = pose.position.y

        dist, index = self.kd_tree_for_tl.query([x, y])
        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if not self.kd_tree_traffilight:
            pts_nd = np.ndarray([len(stop_line_positions), 2])

            for i in range(len(stop_line_positions)):
                x = stop_line_positions[i][0]
                y = stop_line_positions[i][1]
                pts_nd[i][0] = x
                pts_nd[i][1] = y

            self.pts_light = pts_nd
            self.kd_tree_traffilight = scipy.spatial.KDTree(pts_nd)

        #DONE find the closest visible traffic light (if one exists)
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
            x=self.pose.pose.position.x
            y=self.pose.pose.position.y
            dist_light, light_index = self.kd_tree_traffilight.query([x, y])
            self.last_light_index = light_index
            # 200 is enough distance to stop before a traffic light 
            # even if you drive at 110 m/h
            if dist_light<200 and self.kd_tree_for_tl:
                _, index2 = self.kd_tree_for_tl.query(self.pts_light[light_index])
                if index2>car_position:
                    light=True
                    light_index = index2
        
        if light:
            start_time = time.time()
            state = self.get_light_state(light)

            if state == TrafficLight.RED:
                rospy.logerr("wp:{},state:{}".format(light_index,"RED")) 
            elif state == TrafficLight.YELLOW:
                rospy.logerr("wp:{},state:{}".format(light_index,"YELLOW")) 
            elif state == TrafficLight.GREEN:
                rospy.logerr("wp:{},state:{}".format(light_index,"GREEN"))
            else:
                rospy.logerr("wp:{},state:{}".format(light_index,"UNKNOWN"))

            rospy.logerr('--- {} secondes ---'.format(time.time() - start_time))
            return light_index, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
