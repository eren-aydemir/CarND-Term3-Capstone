This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).


### Team Members

Our team is really diverse and spread all over the world. For a complex project like this it is a great challenge but very worth it.

- Onkar Todakar (Sunnyvale, CA) [@onkartodakar](https://github.com/onkartodakar)
  Udacity account : onkartodakar@gmail.com
- Guangjian Yang (Beijing, China) [@goo00d](https://github.com/goo00d)
  Udacity account : y58j32@163.com
- Oliver Park (Seoul, South Korea) [@OliverPark](https://github.com/OliverPark)
  Udacity account : chanho16.park@gmail.com
- Eren Aydemir (Istanbul, Turkey) [@eren-aydemir](https://github.com/eren-aydemir)
  Udacity account : eren.aydemir@gmail.com
- Hernán Foffani (Madrid, Spain) [@hfoffani](https://github.com/hfoffani)
  Udacity account : hfoffani@gmail.com


### Implementation Description

There are three main components in the solution: Drive By Wire, Waypoint Updater, and Traffic Light Detector. An independent ROS  node implements each of these three elements. A description of each one follows.

#### Drive By Wire (DBW)

A PID Controller implements the DBW component (proportional-integral-derivative controller). This mechanism is used extensively in industrial control systems where an accurate and optimised automatic control is required.

The PID controls the throttle and the brakes. We fine-tuned the parameters by trial and error to find the set that gave us the best results. The comparison between the output of the PID controller (the proposed speed) and the current velocity provides us with a hint to apply either throttle or breaks: if the car is driving slower than the proposed speed we accelerate otherwise we break.

Other details in the implementation help to reduce jerking. For instance, we use a Low-Pass filter to smooth steering angle movement.

#### Waypoint Updater

We use a /k-dimensional tree/ to find which waypoint index corresponds to the current position and publish a subset of the next 300 waypoints to the subscribers of this node.

This node also receives notifications from the Traffic Light Detector. If there is a red light ahead, it finds its position in waypoint index and deaccelerate the speed, so the car stops at that point.

#### Traffic Light Detector

To interpret traffic lights, we use Tensorflow's Object Detection sets of algorithms. It subscribes to the car's camera to receive a flow of images. The object detection analyzes each image and returns a set of boxes where there are traffic lights. The node only cares about image sections where the probability is significant, and it publishes either the color of the light or none if there are no signals.

The library provides several pre-trained models that can detect dozens of different objects in an image. We retrained them, like Transfer Learning, using a traffic-light dataset (found in Udacity's forums) and modified the model to identify just four classes (red, yellow, green, and no-lights). There are two different models one for the simulator and another for real images.

We have tried a the Inception and MobileNet algorithms before we settle for the Faster R-CNN. It is a relatively new (2015) that has a right balance between speed and accuracy. In our hardware, we measured over 7 FPS (frames-per-second). We expect it to be even faster in Udacity's autonomous car.

The implementation allows splitting the inference part (what the detector does) from the rest of the steps (build the datasets and training the model) which simplifies a lot the fine-tuning process.

A separate project takes care of all the image data munging, training, testing, exporting the model and benchmarking. Its output is the exported files with the .pb extension which is then copied into this project.


### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
