from yaw_controller import YawController
from lowpass import *
import math
from pid import *
import rospy
import time
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, accel_limit, throttle_limit,tor_limit):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.tau = 0.2
        self.ts = 0.2

        self.lowpass_steer = LowPassFilter(self.tau,self.ts)
        kp = 0.001
        ki = 0
        kd = 0.01
        self.throttle_limit = throttle_limit
        self.pid = PID(kp,ki,kd,-self.throttle_limit,self.throttle_limit)
        self.lastTime = None
        self.accel_limit = accel_limit
        self.accu_time = 0
        self.break_state = False
        self.last_throttle = 0
        self.tor_limit = tor_limit
        self.pid_b = PID(tor_limit*0.6, 0, tor_limit*0.04, -self.tor_limit, self.tor_limit)
        self.last_brake = 0
    def control(self, proposed_v,proposed_angular_v,current_v,dbw_enable):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enable:
            return 0, 0, 0
        if not self.lastTime:
            self.lastTime = time.time()-0.02
        steer = self.yaw_controller.get_steering(proposed_v, proposed_angular_v, current_v)
        steer = self.lowpass_steer.filt(steer)
        currentTime = time.time()
        sample_time = currentTime - self.lastTime
        error = proposed_v - current_v
        if proposed_v-current_v<-1.0 or proposed_v<=1.0:
            self.break_state = True
        if proposed_v - current_v>1.0 and proposed_v>1.0:
            self.break_state = False
        #rospy.logerr("{}:{}:{}".format(current_v, proposed_v, steer))
        throttle = self.last_throttle
        brake = self.last_brake
        val = 0
        if not self.break_state:
            #if sample_time>0.2:
            #if current_v>proposed_v-tol or current_v<proposed_v-4*tol:
            val = self.pid.step(error,sample_time)
            #rospy.logerr("sample_time {} error:{}".format(sample_time, error))
            throttle += val
            throttle = max(0.0,min(self.throttle_limit,throttle))
            brake = 0
            self.lastTime = currentTime
            self.pid_b.reset()
        else:
            val= self.pid_b.step(error,sample_time)
            brake-=val
            brake = max(0, min(brake, self.tor_limit))
            self.pid.reset()
            throttle = 0
            self.lastTime = currentTime

        self.last_throttle = throttle
        self.last_brake = brake

        #rospy.logerr("propose_v:{} current_v:{} error:{} val:{} throttle:{} brake:{} steer:{}".format(proposed_v,current_v,error,val,throttle,brake,steer))
        return throttle,brake,steer
    def init(self):
        self.lowpass_steer = LowPassFilter(self.tau, self.ts)
        self.pid.reset()

