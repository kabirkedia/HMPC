#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

from dataclasses import dataclass, field
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.spatial.transform import Rotation

import cvxpy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

# TODO: import as you need
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import time


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering, acceleration]
    TK: int = 2  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 20.0])
        # default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 5.0])
        # default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 30.0, 5.5])
        # default_factory=lambda: np.diag([100.0, 100.0, 13.0, 100.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 13.0, 5.5])
        # default_factory=lambda: np.diag([100.0, 100.0, 13.0, 100.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    

# class def for tree nodes
# It's up to you if you want to use this
class TreeNode(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class MPC_RRT(Node):
    def __init__(self):
        super().__init__('rrt')
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "pf/pose/odom"     # TBD: will need to be updated
        scan_topic = "/scan"
        drive_topic = "/drive"
        og_topic = "/dynamic_map"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(      # TBD: will need to be updated
            # PoseStamped,
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.og_pub_ = self.create_publisher(OccupancyGrid, og_topic, 1)

        # debugging
        self.tree_nodes_pub_ = self.create_publisher(Marker, '/tree_nodes', 10)
        self.path_pub_ = self.create_publisher(Path, '/found_path', 10)
        self.marker1_pub_ = self.create_publisher(Marker,'/goal_waypoint', 1)
        self.marker2_pub_ = self.create_publisher(Marker,'/local_target', 1)
        self.marker3_pub_ = self.create_publisher(Marker,'/goal_grid', 1)

        # class attributes

        # obstacle detection 
        self.ranges = [0]
        self.step = 0
        self.angle_max = 0
        self.obstacle_detected = False
        self.obstacle_range = 50    # number of lidar scans 
        self.obstacle_threshold = 2.5   # distance to obs

        # occupancy grid attributes
        self.og_height = 2.0            # m
        self.og_width = 3.0             # m
        self.og_resolution = 0.05       # m
        self.kernel_size = 9

        # odometry attributes (global - map frame)
        self.x_current = 0.0
        self.y_current = 0.0
        self.heading_current = 0.0

        # global planner parameters
        self.x_current_goal = 0.0       
        self.y_current_goal = 0.0  
        self.waypoints = np.genfromtxt("/sim_ws/src/lab7/f1tenth_lab7/waypoints/practice2_waypoints.csv", delimiter = ',')
        self.rrt_waypoints = self.waypoints[:, 0 : 2]    

        # physical car attributes
        self.base_to_lidar = 0.27275    # m
        self.edge_to_lidar = 0.10743    # m

        # RRT parameters
        self.max_rrt_iterations = 1000
        self.lookahead_distance = 1.5   # m
        self.steer_range = 0.3          # m``
        self.goal_tolerance = 0.2       # m
        self.collision_checking_points = 20

        # sampling parameters
        self.sample_bias = 0.7
        self.std_deviation = 0.5

        # pure pursuit parameters
        self.clamp_angle = 30.0         # deg
        self.steering_gain = 1.0

        # initialize occupancy grid
        self.occupancy_grid = OccupancyGrid()
        self.init_occupancy_grid()

        # visualization
        self.ref_path_vis_pub_ = self.create_publisher(Marker, "/ref_path_vis", 1)
        self.pred_path_vis_pub_ = self.create_publisher(Marker, "/pred_path_vis", 1)
        self.waypoint_pub_ = self.create_publisher(MarkerArray, "waypoints", 1)

        # MPC params
        self.config = mpc_config()
        self.odelta = None
        self.oa = None
        self.init_flag = 0
        self.velocity_gain = 3.0

        # initialize MPC problem
        self.mpc_prob_init()


    '''MPC functions'''
    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis = 1)), Rd_block)
        
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints += [cvxpy.vec(cvxpy.abs(cvxpy.diff(self.uk[1, :]))) <= self.config.MAX_DSTEER * self.config.DTK]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints += [self.xk[:, 0] == self.x0k]
        constraints += [self.xk[2, :] >= self.config.MIN_SPEED]
        constraints += [self.xk[2, :] <= self.config.MAX_SPEED]
        constraints += [cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL]
        constraints += [cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER]
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = max(abs(state.v), 0.1) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj
    
    
    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict


    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state


    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C


    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov


    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
     

    '''RRT functions'''
    def init_occupancy_grid(self):
        """
        Initialize occupancy grid 
        """
        rows = int(self.og_height // self.og_resolution)
        cols = int(self.og_width // self.og_resolution)
        self.occupancy_grid.header.frame_id = "ego_racecar/base_link"
        self.occupancy_grid.info.width = cols
        self.occupancy_grid.info.height = rows
        self.occupancy_grid.info.resolution = self.og_resolution
        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        data = np.full((rows, cols), -1, np.int8)
        self.occupancy_grid.data = data.flatten().tolist()
        # self.occupancy_grid.info.origin.position.x = -self.base_to_lidar
        self.occupancy_grid.info.origin.position.x = self.base_to_lidar
        self.occupancy_grid.info.origin.position.y = -(rows // 2) * self.og_resolution
        self.occupancy_grid.info.origin.position.z = 0.0
        
        self.og_pub_.publish(self.occupancy_grid)

    
    def convert_map_to_og(self, x_map, y_map):
        """
        Converts map coordinates to laser frame coordinates
        """
        x_base_rot = x_map - self.x_current
        y_base_rot = y_map - self.y_current
        angle = -self.heading_current
        rot_matrix = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]
        [x_base, y_base] = np.matmul(rot_matrix, [x_base_rot, y_base_rot])
        y_grid = y_base
        x_grid = x_base - self.base_to_lidar

        return (x_grid, y_grid)


    def convert_og_to_map(self, x_grid, y_grid):
        """
        Converts laser frame coordinates to map frame
        """
        # convert from laser to base link frame
        x_base = x_grid + self.base_to_lidar
        y_base = y_grid
        # rotate to parallel to map frame
        rot_matrix = [[np.cos(self.heading_current), -np.sin(self.heading_current)],
                        [np.sin(self.heading_current), np.cos(self.heading_current)]]
        [x_base_rot, y_base_rot] = np.matmul(rot_matrix, [x_base, y_base])

        # translate by odom
        x_map = x_base_rot + self.x_current
        y_map = y_base_rot + self.y_current

        return (x_map, y_map)
    

    def is_occupied(self, x_grid, y_grid):
        """
        Checks if lidar coordinate x, y is occupied in occupancy grid
        """
        # get corresponding cell in occupany grid
        row = int(self.occupancy_grid.info.height // 2 + y_grid // self.og_resolution)
        col = int(x_grid // self.og_resolution)

        if (row < 0 or col < 0):
            return False

        if (row >= self.occupancy_grid.info.height or col >= self.occupancy_grid.info.width):
            return False
        
        og_index = int(row * self.occupancy_grid.info.width + col)

        if (self.occupancy_grid.data[og_index] > 0):
            return True
        else:
            return False
    

    def get_next_goal(self):
        """
        Gets next global planner waypoint to get to using RRT
        """
        best_index = -1
        best_goal_distance = 10000.0
        for i in range(len(self.rrt_waypoints)):
            global_x = self.rrt_waypoints[i][0]
            global_y = self.rrt_waypoints[i][1]
            (global_x_grid, global_y_grid) = self.convert_map_to_og(global_x, global_y)

            if (global_x_grid <= 0.0):
                # goal behind car, skip
                continue

            goal_dist = np.abs(self.lookahead_distance - np.sqrt(global_x_grid**2 + global_y_grid**2))

            if (goal_dist < best_goal_distance):
                # make sure it is not an occupied point
                # if (self.is_occupied(global_x_grid, global_y_grid)):
                #     continue

                best_goal_distance = goal_dist
                best_index = i

        x = self.rrt_waypoints[best_index][0]
        y = self.rrt_waypoints[best_index][1]
        
        (x_grid, y_grid) = self.convert_map_to_og(x, y)
        
        p_map = self.display_marker("map", 1.0, 0.0, 0.0, [x, y])
        self.marker1_pub_.publish(p_map)
        
        if (x_grid <= 0):
            print("goal behind car!!!")


        return (self.rrt_waypoints[best_index][0], self.rrt_waypoints[best_index][1])
        

    def get_speed(self, angle):
        abs_angle = np.abs(angle)
        if abs_angle >= np.deg2rad(15):
            speed = 0.75
        elif abs_angle >= np.deg2rad(10):
            speed = 1.25
        elif abs_angle >= np.deg2rad(5):
            speed = 1.5
        else:
            speed = 2.0
        return speed
    

    def inflate_obstacles(self, kernel):

        height = self.occupancy_grid.info.height
        width = self.occupancy_grid.info.width

        # Get kernel dimensions
        k_height, k_width = kernel.shape

        # Compute padding for the input image
        pad_height = k_height // 2
        pad_width = k_width // 2

        og_grid_data = np.array(self.occupancy_grid.data)
        og_grid_data = og_grid_data.reshape((height, width))

        # Create an empty output image
        dilated_image = np.zeros_like(og_grid_data)

        # Pad the input image
        padded_image = np.pad(og_grid_data, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # # Apply dilation
        for y in range(height):
            for x in range(width):
                neighborhood = padded_image[y:y + k_height, x:x + k_width]
                dilated_image[y, x] = np.max(neighborhood * kernel)
        
        self.occupancy_grid.data = dilated_image.flatten().tolist()  


    '''Visualization functions'''
    def display_marker(self, frame, r, g, b, current_waypoint):
        marker = Marker()
        marker.header.frame_id = frame
        marker.ns = "marker"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = 2
        marker.action = 0
        marker.pose.position.x = current_waypoint[0]
        marker.pose.position.y = current_waypoint[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.z = 0.1
        marker.scale.y = 0.25
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker
    

    def visualize_tree(self, tree):
        tree_nodes = Marker()
        tree_nodes.header.frame_id = "map"
        tree_nodes.ns = "marker"
        tree_nodes.header.stamp = self.get_clock().now().to_msg()
        tree_nodes.id = 1
        tree_nodes.type = 8
        tree_nodes.action = 0
        tree_nodes.scale.x = 0.1
        tree_nodes.scale.z = 0.1
        tree_nodes.scale.y = 0.1
        tree_nodes.color.a = 1.0
        tree_nodes.color.r = 0.0
        tree_nodes.color.g = 0.0
        tree_nodes.color.b = 1.0

        for node in tree:
            point = Point()
            point.x = node.x
            point.y = node.y
            point.z = 0.0
            tree_nodes.points.append(point)

        self.tree_nodes_pub_.publish(tree_nodes)
        tree_nodes.points.clear()

    
    def visualize_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for node in path:
            loc = PoseStamped()
            loc.header.stamp = self.get_clock().now().to_msg()

            loc.header.frame_id = "map"
            loc.pose.position.x = node.x
            loc.pose.position.y = node.y
            loc.pose.position.z = 0.00
            path_msg.poses.append(loc)

        self.path_pub_.publish(path_msg)

    
    def visualize_mpc_path(self, ox, oy):
        """
        A method used simply to visualze the the predicted trajectory 
        for the mpc control problem output.

        Inputs:
            ox: the computed x positions from the mpc problem
            oy: the computed y positions from the mpc problem
        """

        mpc_path_vis = Marker()
        mpc_path_vis.header.frame_id = "map"
        mpc_path_vis.color.a = 1.0
        mpc_path_vis.color.r = 0.0
        mpc_path_vis.color.g = 1.0
        mpc_path_vis.color.b = 0.0
        mpc_path_vis.type = Marker.LINE_STRIP
        mpc_path_vis.scale.x = 0.1
        mpc_path_vis.id = 1000

        for i in range(len(ox)):
            mpc_path_vis.points.append(Point(x=ox[i], y=oy[i], z=0.0))

        self.pred_path_vis_pub_.publish(mpc_path_vis)


    def visualize_ref_traj(self, ref_traj):
        """
        A method used simply to visualze the computed reference trajectory
        for the mpc control problem.

        Inputs:
            ref_traj: reference trajectory ref_traj, reference steering angle
                      [x, y, v, yaw]
        """
        ref_strip = Marker()
        ref_strip.header.frame_id = "map"
        ref_strip.ns = "ref_traj"
        ref_strip.id = 10
        ref_strip.type = Marker.LINE_STRIP
        ref_strip.action = Marker.ADD
        ref_strip.scale.x = 0.2
        ref_strip.color.a = 0.4
        ref_strip.color.r = 1.0
        ref_strip.color.g = 0.0
        ref_strip.color.b = 1.0


        # make a list of points from the ref_traj
        ref_strip.points.clear()
        for i in range(ref_traj.shape[1]):
            # p = Point(ref_traj[0, i], ref_traj[1, i])
            p = Point()
            p.x = ref_traj[0, i]
            p.y = ref_traj[1, i]
            ref_strip.points.append(p)

        self.ref_path_vis_pub_.publish(ref_strip)


    def visualize_waypoints(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.POINTS
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.id = 0
        marker.points = [Point(x=x, y=y, z=0.0) for x, y, _, _ in self.waypoints]

        marker_array = MarkerArray()
        marker_array.markers = [marker]
        self.waypoint_pub_.publish(marker_array)


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
        """
        # for obs detection:
        self.step = scan_msg.angle_increment
        self.angle_max = scan_msg.angle_max
        self.ranges = np.array(scan_msg.ranges)

        self.obstacle_detected = False
        if (np.min(len(self.ranges) > 1 and 
            self.ranges[int(self.angle_max / self.step) - self.obstacle_range: int(self.angle_max / self.step) + self.obstacle_range]) < self.obstacle_threshold):
            self.obstacle_detected = True

        # obstacle detected, switch to RRT - update occupancy grid
        if self.obstacle_detected:

            ranges = np.array(scan_msg.ranges)
            rows = self.occupancy_grid.info.height
            cols = self.occupancy_grid.info.width

            proc_ranges = np.copy(ranges)
            proc_ranges[proc_ranges < scan_msg.range_min] = scan_msg.range_min
            proc_ranges[proc_ranges > scan_msg.range_max] = scan_msg.range_max
            proc_ranges[np.isnan(proc_ranges) | np.isinf(proc_ranges)] = scan_msg.range_max

            # Create meshgrid of row and col indices
            col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))

            # Calculate x and y coordinates for each cell
            x_cell = col_indices * self.og_resolution + (self.og_resolution / 2)
            y_cell = (row_indices - (rows / 2)) * self.og_resolution + (self.og_resolution / 2)

            # Calculate distance to each cell
            distance_to_cell = np.sqrt(x_cell**2 + y_cell**2)

            # Calculate angle to each cell
            angle_to_cell = np.arctan2(y_cell, x_cell)

            # Find closest index in scan_msg for each cell
            closest_index = ((angle_to_cell - scan_msg.angle_min) / scan_msg.angle_increment).astype(int)

            # Get distance to object for each cell
            distance_to_obj = proc_ranges[closest_index]

            # Create occupancy grid data
            occupancy_grid_data = np.where(distance_to_cell >= distance_to_obj, 100, 0)

            # Flatten the occupancy grid data and assign it to self.occupancy_grid.data
            self.occupancy_grid.data = occupancy_grid_data.flatten().tolist()

            kernel = np.ones((self.kernel_size, self.kernel_size))
            self.inflate_obstacles(kernel)

        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.og_pub_.publish(self.occupancy_grid)
        self.visualize_waypoints()


    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        self.x_current = pose_msg.pose.pose.position.x
        self.y_current = pose_msg.pose.pose.position.y

        # convert rotation quaternion to heading angle
        q = [pose_msg.pose.pose.orientation.w, pose_msg.pose.pose.orientation.x,  
             pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z]
        sin_angle = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cos_angle = 1.0 - 2.0 * (q[2]**2 + q[3]**2)
        self.heading_current = np.arctan2(sin_angle, cos_angle)
        
        # if obstacle detected, switch to RRT
        if (self.obstacle_detected and len(self.ranges) > 1):
            # TBD: will need to be updated for running on car
            # update next goal global waypoint
            (self.x_current_goal, self.y_current_goal) = self.get_next_goal()
            (goal_x_grid, goal_y_grid) = self.convert_map_to_og(self.x_current_goal, self.y_current_goal)
            dist_to_goal = (self.x_current - self.x_current_goal)**2 + (self.y_current - self.y_current_goal)**2

            if (self.is_occupied(goal_x_grid, goal_y_grid)):
                self.goal_tolerance = 0.9
                self.sample_bias = 0.0
            else:
                self.goal_tolerance = 0.2
                self.sample_bias = 0.7

            # debugging
            # p_grid = self.display_marker("ego_racecar/laser_model", 0.0, 1.0, 0.0, [goal_x_grid, goal_y_grid])
            # self.marker3_pub_.publish(p_grid)
            # p_map = self.display_marker("map", 1.0, 0.0, 0.0, [self.x_current_goal, self.y_current_goal])
            # self.marker1_pub_.publish(p_map)

            # define starter node
            start_node = TreeNode()
            start_node.x = self.x_current
            start_node.y =  self.y_current
            start_node.is_root = True
            start_node.parent = 0
            start_node.cost = 0.0

            # initialize tree
            tree = [start_node]

            for iter in range(self.max_rrt_iterations):
                sampled_point = self.sample()
                dist_to_sample = (self.x_current - sampled_point[0])**2 + (self.y_current - sampled_point[1])**2

                (sample_x_grid, sample_y_grid) = self.convert_map_to_og(sampled_point[0], sampled_point[1])
                if ((self.is_occupied(sample_x_grid, sample_y_grid)) or (dist_to_sample > dist_to_goal)):
                    continue

                nearest_indx = self.nearest(tree, sampled_point)
                new_node = self.steer(tree[nearest_indx], sampled_point)
                
                (new_x_grid, new_y_grid) = self.convert_map_to_og(new_node.x, new_node.y)
                if (self.is_occupied(new_x_grid, new_y_grid)):
                    continue

                new_node.parent = nearest_indx

                if (not self.check_collision(tree[nearest_indx], new_node)):
                    
                    tree.append(new_node)
                    if (self.is_goal(new_node, self.x_current_goal, self.y_current_goal)):
                        # close enough to goal
                        path = self.find_path(tree, new_node)
                        self.track_path(path)
                        # self.visualize_tree(tree)
                        self.visualize_path(path)
                        break

        else:
            vehicle_state = State()
            vehicle_state.x = pose_msg.pose.pose.position.x
            vehicle_state.y = pose_msg.pose.pose.position.y
            vehicle_state.v = np.sqrt(pose_msg.twist.twist.linear.x**2 + pose_msg.twist.twist.linear.y**2)
            q = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
            quat = Rotation.from_quat(q)
            euler = quat.as_euler("zxy", degrees=False)
            yaw = euler[0]
            yaw = (yaw + 2 * np.pi) % (2 * np.pi)
            
            vehicle_state.yaw = yaw

            # TODO: Calculate the next reference trajectory for the next T steps
            #       with current vehicle pose.
            #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
            ref_x = self.waypoints[:, 0]
            ref_y = self.waypoints[:, 1]
            ref_yaw = self.waypoints[:, 2]

            ref_yaw = (ref_yaw + 2 * np.pi) % (2 * np.pi)

            ref_v = self.waypoints[:, 3] * self.velocity_gain

            ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)

            self.visualize_ref_traj(ref_path)

            x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

            # TODO: solve the MPC control problem
            (
                self.oa,
                self.odelta,
                ox,
                oy,
                oyaw,
                ov,
                state_predict,
            ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta)

            self.visualize_mpc_path(ox, oy)

            # TODO: publish drive message.
            steer_output = self.odelta[0]
            speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

            print("MPC control, steering at: ", np.rad2deg(steer_output))
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed_output
            drive_msg.drive.steering_angle = steer_output
            self.drive_pub_.publish(drive_msg)


    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        # x = np.random.uniform(0.0, self.og_width) # lidar frame
        # y = np.random.uniform(- self.og_height / 2, self.og_height / 2)   # lidar frame

        (goal_grid_x, goal_grid_y) = self.convert_map_to_og(self.x_current_goal, self.y_current_goal)

        if np.random.rand() < self.sample_bias:
            # Sample near the goal point using Gaussian distribution
            x = np.random.normal(goal_grid_x, self.std_deviation)
            y = np.random.normal(goal_grid_y, self.std_deviation)
        else:
            # Sample uniformly in the free space
            x = np.random.uniform(0.0, self.og_width)  # lidar frame
            y = np.random.uniform(-self.og_height / 2, self.og_height / 2)  # lidar frame
        
        # if ((not self.is_occupied(x, y)) and (x <= goal_grid_x) and (x >= self.base_to_lidar)):
        #     # convert to map coordinates from grid coordinates

        (x_map, y_map) = self.convert_og_to_map(x, y)

        return (x_map, y_map)
        # else:
        #     return self.sample()


    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        nearest_node = 0
        # min_dist = 10000.0
        # for i in range(len(tree)):
        #     node = tree[i]
        #     sq_dist = (sampled_point[0] - node.x)**2 + (sampled_point[1] - node.y)**2
        #     if (sq_dist < min_dist):
        #         nearest_node = i
        #         min_dist = sq_dist
        sampled_point = np.array(sampled_point)
        tree_points = np.array([(node.x, node.y) for node in tree])

        # Calculate squared distances between sampled point and all tree nodes
        sq_distances = np.sum((tree_points - sampled_point)**2, axis=1)

        # Find the index of the node with the minimum squared distance
        nearest_node = np.argmin(sq_distances)

        return nearest_node


    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        new_node = TreeNode()
        dist = np.sqrt((sampled_point[0] - nearest_node.x)**2 + (sampled_point[1] - nearest_node.y)**2)
        x = sampled_point[0] - nearest_node.x
        y = sampled_point[1] - nearest_node.y

        if (dist < self.steer_range):
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
        else:
            theta = np.arctan2(y, x)
            new_node.x = nearest_node.x + np.cos(theta) * self.steer_range
            new_node.y = nearest_node.y + np.sin(theta) * self.steer_range
    
        # new_node.x = nearest_node.x + min(self.steer_range, dist) * (sampled_point[0] - nearest_node.x) / dist
        # new_node.y = nearest_node.y + min(self.steer_range, dist) * (sampled_point[1] - nearest_node.y) / dist
        return new_node
    

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        collision = False
        x_cell_diff = abs(int((nearest_node.x - new_node.x) / self.collision_checking_points))
        y_cell_diff = abs(int((nearest_node.y - new_node.y) / self.collision_checking_points))

        # dt = 1.0 / max(x_cell_diff, y_cell_diff)
        # t = 0.0
        current_x = nearest_node.x
        current_y = nearest_node.y

        for i in range(self.collision_checking_points):
            # x = nearest_node.x + t * (new_node.x - nearest_node.x)
            # y = nearest_node.y + t * (new_node.y - nearest_node.y)
            current_x += x_cell_diff
            current_y += y_cell_diff

            # convert map to grid coordinates to check if occupied
            # (x_grid, y_grid) = self.convert_map_to_og(x, y)
            (x_grid, y_grid) = self.convert_map_to_og(current_x, current_y)

            if (self.is_occupied(x_grid, y_grid)):
                collision = True
                break

            # t += dt

        return collision


    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        distance = np.sqrt((latest_added_node.x - goal_x)**2 + (latest_added_node.y - goal_y)**2)
        return distance < self.goal_tolerance


    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        current = latest_added_node
        while (not current.is_root):
            path.append(current)
            current = tree[current.parent]
        path.append(current)
        path.reverse()

        (goal_x_grid, goal_y_grid) = self.convert_map_to_og(self.x_current_goal, self.y_current_goal)
        if (not self.is_occupied(goal_x_grid, goal_y_grid)):
            goal_node = TreeNode()
            goal_node.x = self.x_current_goal
            goal_node.y = self.y_current_goal
            path.append(goal_node)
        
        return path
    

    def track_path(self, path):
        """
        Finds node in path just within lookahead distance and follows pure pursuit
        """
        best_index = 0
        closest_distance = 10000.0
        closest_distance_current_pose = 10000.0
        for i in range(len(path)):
            x = path[i].x
            y = path[i].y
            dist = (self.x_current - x) ** 2 + (self.y_current - y) ** 2
            diff_distance = np.abs(self.lookahead_distance - dist)
            
            if (diff_distance < closest_distance):
                closest_distance = diff_distance
                best_index = i
                closest_distance_current_pose = dist
        
        # get next point for pure pursuit using average
        p1 = np.array([path[best_index].x, path[best_index].y])
        avg_target_map = p1
        avg_target_base = self.convert_map_to_og(avg_target_map[0], avg_target_map[1])

        if (self.is_occupied(avg_target_base[0], avg_target_base[1])):
            print("target occupied")

        # debugging
        p_map = self.display_marker("map", 0.0, 0.0, 1.0, [avg_target_map[0], avg_target_map[1]])
        self.marker2_pub_.publish(p_map)

        # calculate curvature/steering angle
        angle = (2 * avg_target_base[1]) / (closest_distance_current_pose ** 2)
        angle = np.clip(angle, -np.deg2rad(self.clamp_angle), np.deg2rad(self.clamp_angle))
        angle = self.steering_gain * angle

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = self.get_speed(angle)

        print("RRT path, steering at angle: ", np.rad2deg(angle))

        self.drive_pub_.publish(drive_msg)


def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


def main(args=None):
    rclpy.init(args=args)
    print("MPC RRT Initialized")
    mpc_node = MPC_RRT()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

