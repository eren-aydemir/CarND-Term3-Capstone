#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import scipy
from scipy import spatial
import numpy as np
import yaml
import time
import copy
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from std_msgs.msg import Int32
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 300 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        # TODO: Add other member variables you need below
        self.waypoints = None
        self.kd_tree = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        pts = self.config['stop_line_positions']
        pts_nd = np.ndarray([len(pts), 2])
        for i in range(len(pts)):
            x = pts[i][0]
            y = pts[i][1]
            pts_nd[i][0] = x
            pts_nd[i][1] = y
        self.pts_light = pts_nd
        self.kd_tree_traffilight = scipy.spatial.KDTree(pts_nd)
        self.light_stop = False
        self.velocity = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))
        self.processed_red = True
        self.processed_green = True
        self.light_index = 0
        self.last_state = None
        self.light_waypoint_index = None
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        if self.waypoints and self.kd_tree:
            lane = Lane()
            lane.header.frame_id = '/uploader'
            lane.header.stamp = rospy.Time(0)
            x=msg.pose.position.x
            y=msg.pose.position.y

            dist,index = self.kd_tree.query([x,y])
            lane.waypoints = self.waypoints[index:index+LOOKAHEAD_WPS]

            dist_light,light_index = self.kd_tree_traffilight.query([x,y])
            _,light_waypoint_index = self.kd_tree.query(self.pts_light[light_index])
            self.light_index = light_index
            if self.light_stop and dist_light<80:
                index2=self.light_waypoint_index-4 #ahead some waypoint to set velocity zero
                if index2>index and index2-index<=LOOKAHEAD_WPS and not self.processed_red:
                    self.decelerate(lane.waypoints[:index2-index])
                    for k in range(index2-index,len(lane.waypoints)):
                        self.set_waypoint_velocity(lane.waypoints,k,0)
                    self.processed_red = True
            elif dist_light>10 and dist_light<30 and light_waypoint_index>index:
                for k in range(len(lane.waypoints)):
                    self.set_waypoint_velocity(lane.waypoints, k, 0.3*self.velocity)
            else:
                if not self.light_stop and not self.processed_green:
                    self.processed_green = True
                    for k in range(len(lane.waypoints)):
                        self.set_waypoint_velocity(lane.waypoints,k,self.velocity)
            self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        if not self.waypoints:
            self.waypoints = waypoints.waypoints
        if not self.kd_tree:
            waypoints_size = len(waypoints.waypoints)
            pts = np.ndarray([waypoints_size,2])
            for i in range(waypoints_size):
                x=waypoints.waypoints[i].pose.pose.position.x
                y=waypoints.waypoints[i].pose.pose.position.y
                pts[i][0]=x
                pts[i][1]=y
            self.kd_tree = scipy.spatial.KDTree(pts)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.light_waypoint_index = msg.data
        if msg.data >=0: #and msg.lights[msg].state in [TrafficLight.RED, TrafficLight.YELLOW]:
            if self.last_state == True:
                return
            self.processed_red = False
            self.light_stop = True
        else:
            if self.last_state == False:
                return
            self.processed_green = False
            self.light_stop = False
        self.last_state = self.light_stop

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)
    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints
    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
