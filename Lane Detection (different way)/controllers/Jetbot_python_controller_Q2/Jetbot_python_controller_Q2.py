"""Jetbot_python_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot , Supervisor
import math
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os, glob
from controller import Display
from controller import CameraRecognitionObject
# create the Robot instance.
# robot = Robot()
#creating Supervisor instance
robot = Supervisor()
mybot_node = robot.getSelf()
rot_field = mybot_node.getField('rotation')

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# define wheel and robot parameters
wheel_height = 0.0078 #0.032
wheel_radius = 0.0299 #0.0153
# wheel_distance = 0.2  # distance between wheels
max_speed = 50
lookahead_distance = 1  # distance to lookahead point

# define initial and final positions and orientations
x_n_1 = -39.5
y_n_1 = 40.35
theta_n_1 = 0

x_f = -45.1
y_f = 45.5
theta_f = 3*math.pi/2

# define lane detection parameters
lane_width = 5  # width of the lane
camera = robot.getDevice('camera')
camera.enable(timestep)
camera_width = camera.getWidth()
camera_height = camera.getHeight()

wheels_speed = 3.5  # constant speed for both wheels

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 70.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True

# get motor and position sensor devices
left_motor = robot.getDevice('left_wheel_hinge')
right_motor = robot.getDevice('right_wheel_hinge')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
left_ps = robot.getDevice('left_position_sensor')
right_ps = robot.getDevice('right_position_sensor')
left_ps.enable(timestep)
right_ps.enable(timestep)
# get IMU device
imu = robot.getDevice('imu')
imu.enable(timestep)
# get display device for debugging
status_display = robot.getDevice('display')

# calculate initial distance and angle to the final position and orientation
dx = x_f - x_n_1
dy = y_f - y_n_1
distance_to_goal = math.sqrt(dx*dx + dy*dy)
angle_to_goal = math.atan2(dy, dx) - theta_n_1
angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))

def convert_hls(image):
    return cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2HLS)
    
def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def get_lane_path(image,lines):
    
    width, height = image.shape[:2]
    
    # Calculate the slope and intercept of the two longest lines to generate the lane path
    if lines is not None:
        slopes = []
        intercepts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            slopes.append(slope)
            intercepts.append(intercept)
        indices = np.argsort(np.abs(slopes)).astype(int)
        lane_slopes = [slopes[i] for i in indices[-2:]]
        lane_intercepts = [intercepts[i] for i in indices[-2:]]
        lane_y1 = int(height/3)
        lane_y2 = height
        lane_x1 = int((lane_y1 - lane_intercepts[0])/lane_slopes[0])
        lane_x2 = int((lane_y2 - lane_intercepts[0])/lane_slopes[0])
        path = [(lane_x1, lane_y1), (lane_x2, lane_y2)]
        return path
    else:
        return None

def pure_pursuit_controller(x_n, y_n, theta_n, path,left_lane,right_lane):
    lookahead_distance = 0.5  
    # Calculate the curvature of the path at the lookahead point
    if(left_lane!=None)and(right_lane!=None):
        lookahead_point = Lookahead_point([x_n, y_n], theta_n,left_lane,right_lane)
    elif(left_lane!=None)and(right_lane==None):
        left_speed = wheels_speed
        right_speed = wheels_speed/3
        # lookahead_point = [x_n+1,y_n+0.25]
        print("right")
        return left_speed, right_speed
    elif(left_lane==None)and(right_lane!=None):
        left_speed = wheels_speed/3
        right_speed = wheels_speed
        print("left")
        return left_speed, right_speed
    else:
        lookahead_point = [x_n,y_n]
        
    curvature = calculate_curvature(lookahead_point[0], lookahead_point[1], x_n, y_n, theta_n)
    
    # Calculate the desired speed based on the curvature
    speed = max_speed / (1.0 + abs(curvature))
    
    # Calculate the desired wheel speeds based on the curvature and speed
    left_speed = speed * (2 - curvature * wheel_height) / 1 * wheel_radius
    right_speed = speed * (2 + curvature * wheel_height) / 1 * wheel_radius
    # print(lookahead_point, "\t",left_speed, right_speed)
    return left_speed, right_speed
   
def calculate_curvature(x, y, x_n, y_n, theta_n):
    dx = x - x_n
    dy = y - y_n
    numerator = 2 * dx
    denominator = math.sqrt(dx**2 + dy**2)
    # print("denominator" ,denominator)
    # try:
    curvature = numerator / denominator**2
    # except:
        # curvature = 0.0001
    # print("curvature" ,curvature)
    return curvature
def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[1] # bottom of the image
    # print("y1 ", y1)
    y2 = y1*0.33         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line
    
def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                line_length = np.sqrt((x2-x1)**2+(y2-y1)**2)
                # print("line length = ",line_length)
                # if(line_length < 120): 
                    # x2+=100
                    # y2+=100
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        
    return image

def detect_lane(image):
    ylw_wte_image = select_white_yellow(image).astype(np.uint8)
    # Convert the image to grayscale
    gray = cv2.cvtColor(ylw_wte_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection to identify edges
    edges = cv2.Canny(blur, 50, 150)
    
    # Mask the region of interest
    mask = np.zeros_like(edges)
    width, height = image.shape[:2]
    # print("h , w: ",height, width)
    vertices = np.array([[(height, -15),(height/2.8, width/2.7), (height/2.8, width/1.7), (height,  width+15)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=500)
    result = draw_lines(image, lines)
    if result is not None:
            ir = status_display.imageNew(result.tolist(), Display.RGB, camera_width, camera_height)
            status_display.imagePaste(ir, 0, 0, False)
            status_display.imageDelete(ir)
    return result,lines

# Define a function to find the lookahead point
def Lookahead_point(robot_position, heading,left_lane,right_lane):
    # Find the closest point on the left and right lane lines to the robot
    left_closest = closest_point_on_line(left_lane, robot_position)
    right_closest = closest_point_on_line(right_lane, robot_position)

    # Choose the lane line that is on the same side as the robot's heading
    if heading > 0:
        target_line = right_lane
    else:
        target_line = left_lane

    # Calculate the slope and y-intercept of the target line
    m = (target_line[1][1] - target_line[0][1]) / (target_line[1][0] - target_line[0][0])
    b = target_line[0][1] - m * target_line[0][0]

    # Calculate the x-coordinate of the lookahead point
    x = (m * robot_position[0] - robot_position[1] + b) / (m ** 2 + 1)

    # Calculate the y-coordinate of the lookahead point
    y = m * x +b
    return [x,y]
    
# Define a function to find the closest point on a line to a given point
def closest_point_on_line(line, point):
    # Calculate the slope and y-intercept of the line
    m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
    b = line[0][1] - m * line[0][0]

    # Calculate the x-coordinate of the point on the line closest to the given point
    x = (point[0] + m * point[1] - m * b) / (m ** 2 + 1)

    # Calculate the y-coordinate of the point on the line closest to the given point
    y = m * x + b

    return (x, y)
    
def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                    
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
                    
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    lw_total = np.sum(left_weights)
    rw_total = np.sum(right_weights)
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    try:
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    except:
      print("RuntimeWarning: divide by zero encountered in double_scalars")
      pass

def ekf_slam(xEst, PEst, u, z):
    # Predict
    S = STATE_SIZE
    G, Fx = jacob_motion(xEst[0:S], u)
    xEst[0:S] = motion_model(xEst[0:S], u)
    PEst[0:S, 0:S] = G.T @ PEst[0:S, 0:S] @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)

    # Update
    for iz in range(len(z[:, 0])):  # for each observation
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

        nLM = calc_n_lm(xEst)
        if min_id == nLM:
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        lm = get_landmark_position_from_state(xEst, min_id)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    
    return xTrue, z, xd, ud

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    # v = u[0]
    # w = u[1]
    # theta = x[2]
    # x_dot = v * math.cos(theta)
    # y_dot = v * math.sin(theta)
    # theta_dot = w
    # return np.array([[x_dot], [y_dot], [theta_dot]])
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y)

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H


def jacob_h(q, delta, x, i):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

      
            
def main():
    
    status_display = robot.getDevice('display')
    
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()
    
    imu = robot.getDevice('imu')
    imu.enable(timestep)
    
    wheels_speed = 3.5
    
    old_ps=[0, 0]
    new_ps=[0, 0]
    
    x_n_1=-39.5
    y_n_1=40.35
    theta_n_1=0
    
    x_f = -45.1
    y_f = 45.5
    theta_f = 3*math.pi/2
    
    def image_processing(image):
        final_image = detect_lane(image)##ylw_wte_image.astype(np.uint8))
        return final_image
    
    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[-39.14, 40.50],
                     [-38.57, 40.50],
                     [-38.12, 40.96],
                     [-37.86, 40.96],
                     [-37.86, 41.36],
                     [-37.66, 41.83],
                     [-37.29, 42.24],#[-37.53, 42.24],
                     # [-37.29, 42.81],
                     [-37.53, 42.81],
                     [-37.29, 44],#[-37.53, 44],
                     # [-37.29, 45],
                     [-37.53, 45],
                     [-37.29, 46],
                     # [-37.53, 46],
                     ])
                     
    # State Vector [x y yaw v]'
    # xEst = np.zeros((STATE_SIZE, 1))
    xEst = np.array([[x_n_1],[y_n_1],[theta_n_1]])
    # xTrue = np.zeros((STATE_SIZE, 1))
    xTrue = np.array([[x_n_1],[y_n_1],[theta_n_1]])
    PEst = np.eye(STATE_SIZE)
    # xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning
    xDR = np.array([[x_f],[y_f],[theta_f]])

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        time += DT
        u = calc_input()
        # Read the sensors:
        curr_image = camera.getImageArray()
        curr_image = np.array(curr_image)
        
        #print(curr_image.shape[::-1][1:])
        processed_img,lines = image_processing(curr_image)

        new_ps[0]=left_ps.getValue()*wheel_radius
        new_ps[1]=right_ps.getValue()*wheel_radius
        
        left_wheel_vel = (new_ps[0]-old_ps[0])/(timestep*0.001)
        right_wheel_vel = (new_ps[1]-old_ps[1])/(timestep*0.001)
        
        angles = imu.getRollPitchYaw()
        
        #calculating X_dot, Y_dot of the robot coordinates 
        X_dot = (left_wheel_vel + right_wheel_vel)/2.0
        Y_dot = 0
        
        
        #Calculating the velocities for the world's coordinates 
        x_n_dot_i = math.cos(theta_n_1)*X_dot - math.sin(theta_n_1)*Y_dot
        y_n__dot_i = math.sin(theta_n_1)*X_dot + math.cos(theta_n_1)*Y_dot
        
        
        #solve the differential equations to find the robot's position 
        x_n = x_n_dot_i*timestep*0.001 + x_n_1
        y_n = y_n__dot_i*timestep*0.001 + y_n_1
        theta_n = angles[2]
        
        print(str(x_n)+'     '+str(y_n)+'     '+str(theta_n))
        
        ###### EKF #########
        
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        # print("hx",hxTrue)
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
        ################################
        old_ps[0] = new_ps[0]
        old_ps[1] = new_ps[1]
        
        x_n_1 = x_n
        y_n_1 = y_n
        theta_n_1 = theta_n
        
        trans_field = mybot_node.getPosition()
        rot = rot_field.getSFRotation()
        x_present = trans_field[0]
        y_present = trans_field[1]
        angle = 1.57

        #######code for lane detection######################
        try:
            path = get_lane_path(curr_image,lines)     
            leftline , rightline  = lane_lines(curr_image,lines)
            leftlane, rightlane = average_slope_intercept(lines)
            left_speed,right_speed = pure_pursuit_controller(x_n, y_n, theta_n, path,leftline, rightline)   
        except:
            print("could stop the program")
            left_speed  = wheels_speed
            right_speed = wheels_speed
            if -39 <= x_present <= -37:
                print("Turn 1")
                if (40.25<= y_present <41.75)  :
                    rot_field.setSFRotation([0,0,1,angle/2])
                else:
                    rot_field.setSFRotation([0,0,1,angle])
            if (46.5<= y_present <= 48) :
                print("Turn 2")
                if (-39 <= x_present <= -37.25):
                    rot_field.setSFRotation([0,0,1,3*angle/2])
                else:
                    rot_field.setSFRotation([0,0,1,2*angle])
                
            if -45.5 <= x_present <= -43.45:
                print("Turn 3")
                if (48.0<= y_present <= 48.2):
                    rot_field.setSFRotation([0,0,1,5*angle/2])
                else:
                    rot_field.setSFRotation([0,0,1,3*angle])
                    if trans_field[1] <= 45.5 :
                        wheels_speed = 0    
        
        # ############################################################
        # Process sensor data here.
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        pass
    plt.show()
# Enter here exit cleanup code.
if __name__ == '__main__':
    main()