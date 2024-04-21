#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA


# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering, acceleration]
    TK: int = 3  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 20.0])  # np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 20.0])  # np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 22.0])  # np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 22.0])  # np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.05  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 3.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0 # minimum backward speed [m/s]
    MAX_ACCEL: float = 2.5  # maximum acceleration [m/ss]
    MAX_ITER = 3
    DU_TH = 0.1


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0


class MPC(Node):
    """
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        pose_topic = "/ego_racecar/odom"
        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 0)
        self.lidar_sub_ = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)

        self.waypoint_pub = self.create_publisher(MarkerArray, '/waypoints', 10)
        self.target_pub = self.create_publisher(Marker, '/ref_trajectory', 10)
        self.local_path_viz = self.create_publisher(Marker, '/local_path', 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 0)

        # TODO: get waypoints here
        self.waypoints = np.genfromtxt('/home/kabir/sim_ws/src/mpc/levine.csv', delimiter=",")

        local_traj_path = np.load("/home/kabir/sim_ws/src/mpc/spline_trajs.npy")
        self.local_path = local_traj_path
        self.config = mpc_config()
        self.odelta = None
        self.oa = None
        self.init_flag = True
        self.minRange = 0.2
        self.maxRange = 20.0
        self.disparity_thresh = 1.0

        self.ranges = [0]
        self.angleMax = 0
        self.step = 1e-4

        self.local_vis_msg = Marker()
        self.local_vis_msg.header.frame_id = "/map"
        self.local_vis_msg.type = Marker.POINTS
        self.local_vis_msg.action = Marker.ADD
        self.local_vis_msg.scale.x = 0.1
        self.local_vis_msg.scale.y = 0.1
        self.local_vis_msg.scale.z = 0.1
        self.local_vis_msg.color.g = 1.0
        self.local_vis_msg.color.a = 1.0
        self.local_vis_msg.pose.orientation.w = 1.0
        self.local_vis_msg.lifetime.nanosec = 30
        self.local_vis_msg.points = []
        self.local_vis_msg.color.g = 0.0
        self.local_vis_msg.color.r = 0.0
        self.local_vis_msg.color.b = 0.0

        # initialize MPC problem
        self.mpc_prob_init()

    def lidar_callback(self, data):
        """
        Process each LiDAR scan and Apply Disparity Extender algorithm
        """
        self.step = data.angle_increment
        print(self.step)
        self.angleMax = data.angle_max
        self.ranges = np.array(data.ranges)
        self.islidaron = True
        proc_ranges = self.ranges
        proc_ranges[np.isinf(proc_ranges)] = self.maxRange
        proc_ranges[np.isnan(proc_ranges)] = self.minRange
        disparity = proc_ranges[:-1] - proc_ranges[1:]
        disparity_bool = np.abs(disparity) >= self.disparity_thresh
        disparity_bool_idx = np.where(disparity_bool)[0]

        for idx in disparity_bool_idx:
            min_idx = max(0, idx - 30)
            max_idx = min(idx + 30, proc_ranges.shape[0])
            proc_ranges[min_idx:max_idx] = np.min(proc_ranges[min_idx:max_idx])

        self.ranges = proc_ranges

    def pose_callback(self, pose_msg):

        # TODO: extract pose from ROS msg
        vehicle_state = None
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        currPose = np.array([x, y, 0]).reshape((3, -1))
        quaternion = np.array([pose_msg.pose.pose.orientation.x,
                               pose_msg.pose.pose.orientation.y,
                               pose_msg.pose.pose.orientation.z,
                               pose_msg.pose.pose.orientation.w])
        rot = Rotation.from_quat(quaternion)

        # Convert the quaternion to Euler angles (in radians)
        euler = rot.as_euler('xyz')
        yaw = euler[2]
        yaw = (yaw + 2 * math.pi) % (2 * math.pi)
        v = np.sqrt(pose_msg.twist.twist.linear.x ** 2 + pose_msg.twist.twist.linear.y ** 2)
        vehicle_state = State(x=x, y=y, yaw=yaw, v=v)

        ref_x = self.waypoints[:, 0]
        ref_y = self.waypoints[:, 1]
        ref_yaw = self.waypoints[:, 2]
        ref_yaw = (ref_yaw + 2 * math.pi) % (2 * math.pi)
        print(self.waypoints.shape)
        ref_v = np.ones(self.waypoints.shape[0])

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints

        if (self.init_flag):
            self.init_flag = False
            self.target_ind = np.argmin(np.linalg.norm(self.waypoints[:, :2] - np.array([x, y]).reshape(1, -1), axis=1))
            self.target_ind, _ = self.calc_nearest_index(vehicle_state, ref_x, ref_y, ref_yaw, self.target_ind)
            self.odelta, self.oa = None, None
        self.target_ind, _ = self.calc_nearest_index(vehicle_state, ref_x, ref_y, ref_yaw, self.target_ind)

        if (self.target_ind > len(ref_x) - self.config.TK):
            self.target_ind = 0

        # ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        ref_path, self.target_ind, dref = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v,
                                                                   self.config.dlk,
                                                                   self.target_ind)
        self.publish_ref_traj(pose_msg, ref_path)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        print(ref_path.shape)

        dummy_zeros = np.zeros(self.local_path.shape[1]).reshape(-1, 1)

        # Shape: 101,3
        self.local_vis_msg.points = []
        self.local_vis_msg.colors = []

        best_trajectory_number = 0
        best_closest_idx = 0

        # Shape: [50,2]
        num_idx_to_search = 35
        if self.target_ind + num_idx_to_search < self.waypoints.shape[0]:
            points_to_search = self.waypoints[self.target_ind: self.target_ind + num_idx_to_search, :2]
        else:
            points_to_stack = (self.target_ind + num_idx_to_search) - self.waypoints.shape[0]
            points_to_search = self.waypoints[self.target_ind: self.target_ind + num_idx_to_search, :2]
            points_to_search = np.vstack((points_to_search, self.waypoints[: points_to_stack, :2]))

        obstacle_detect = False
        if np.min(len(self.ranges) > 1 and self.ranges[int(self.angleMax / self.step) - 50: int(self.angleMax / self.step) + 50]) < 2.5:
            obstacle_detect = True
            print("Obstacle Detected")

        if(obstacle_detect and len(self.ranges)>1):
            for i in range(self.local_path.shape[0]):
                self.local_path_1 = self.local_path[i, :, :2]
                rdist, theta_r = self.local_path[i, :, -2], self.local_path[i, :, -1]
                theta_index = int(self.angleMax / self.step) + (theta_r / self.step).astype(int)
                is_occ = np.max(self.ranges[theta_index] < rdist)
                world_local_points_1 = rot.apply(np.hstack((self.local_path_1, dummy_zeros))) + currPose.T

                if is_occ:
                    c = ColorRGBA()
                    c.r = 1.0
                    c.b = 0.0
                    c.g = 0.0
                    c.a = 1.0
                else:
                    c = ColorRGBA()
                    c.r = 0.0
                    c.b = 1.0
                    c.g = 0.0
                    c.a = 1.0
                    lastp = world_local_points_1[-1, :][:2].reshape(1, -1)  # Shape:(1,2)
                    closest_idx = np.argmin(np.linalg.norm(lastp - points_to_search, axis=1))
                    if closest_idx >= best_closest_idx:
                        best_closest_idx = closest_idx
                        best_trajectory_number = i

                for j in range(self.local_path_1.shape[0]):
                    p = Point()
                    p.x = world_local_points_1[j, 0]
                    p.y = world_local_points_1[j, 1]
                    p.z = 0.0

                    self.local_vis_msg.points.append(p)
                    self.local_vis_msg.colors.append(c)

            print("Best Trajectory number:",best_trajectory_number)
            self.local_path_1 = self.local_path[best_trajectory_number, :, :2]  # Shape: 101,
            world_local_points_1 = rot.apply(np.hstack((self.local_path_1, dummy_zeros))) + currPose.T
            spline_yaw = self.local_path[best_trajectory_number, :, 2] + yaw
            c = ColorRGBA()
            c.r = 0.937
            c.b = 0.258
            c.g = 0.960
            c.a = 1.0

            for j in range(self.local_path_1.shape[0]):
                p = Point()
                p.x = world_local_points_1[j, 0]
                p.y = world_local_points_1[j, 1]
                p.z = 0.0
                self.local_vis_msg.points.append(p)
                self.local_vis_msg.colors.append(c)

            self.local_path_viz.publish(self.local_vis_msg)

            local_x = world_local_points_1[:, 0]
            local_y = world_local_points_1[:, 1]
            idx_to_sample = np.linspace(0, world_local_points_1.shape[0] - 1, self.config.TK + 1, dtype=int)

            local_ref = ref_path
            local_ref[0, :] = local_x[idx_to_sample]
            local_ref[1, :] = local_y[idx_to_sample]

            local_spline_yaw = spline_yaw[idx_to_sample]
            local_spline_yaw[local_spline_yaw < 0] = local_spline_yaw[local_spline_yaw < 0] + 2 * math.pi

            if (abs(vehicle_state.yaw - local_spline_yaw[0]) > math.pi):
                if (vehicle_state.yaw < local_spline_yaw[0]):
                    vehicle_state.yaw += 2 * math.pi
                else:
                    print("hey you out there in the cold, ")
                    local_spline_yaw[0] += 2 * math.pi

            for i in range(1, local_spline_yaw.shape[0]):
                if local_spline_yaw[i] < 1.0 and local_spline_yaw[i - 1] > 6.0:
                    local_spline_yaw[i] += 2 * math.pi

            for i in range(self.config.TK - 1, -1, -1):
                if (local_spline_yaw[i] - local_spline_yaw[i + 1]) < -math.pi:
                    local_spline_yaw[i] += 2 * math.pi

            if (vehicle_state.yaw - local_spline_yaw[0]) < -math.pi:
                vehicle_state.yaw += 2 * math.pi

            local_ref[3, :] = local_spline_yaw
            ref_path = local_ref

        self.local_path_viz.publish(self.local_vis_msg)

        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(ref_path, x0, dref, self.oa, self.odelta)

        # TODO: solve the MPC control problem
        # (
        #     self.oa,
        #     self.odelta,
        #     ox,
        #     oy,
        #     oyaw,
        #     ov,
        #     state_predict,
        # ) = self.iterative_linear_mpc_control(ref_path, x0, dref, self.oa, self.odelta)

        # TODO: publish drive message.
        if self.oa is not None:
            di, ai = self.odelta[0], self.oa[0]
            msg = AckermannDriveStamped()
            msg.drive.acceleration = float(ai)
            msg.drive.steering_angle = float(di)
            self.old_input = di
            msg.drive.speed = float(ref_v[self.target_ind]) * 0.75
            self.drive_pub_.publish(msg)
            print(f'Driving command published speed={msg.drive.speed}, angle={msg.drive.steering_angle}')
            self.publish_waypoints()

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
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
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
        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) +
                        self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_]

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
        constraints += [self.xk[:, 0] == self.x0k,  # Initial state constraint
                        self.config.MIN_SPEED <= self.xk[2, :],  # Lower bound on speed
                        self.xk[2, :] <= self.config.MAX_SPEED,  # Upper bound on speed
                        -self.config.MAX_STEER <= self.uk[1, :],  # Lower bound on steering angle
                        self.uk[1, :] <= self.config.MAX_STEER,  # Upper bound on steering angle
                        -self.config.MAX_ACCEL <= self.uk[0, :],  # Lower bound on acceleration
                        self.uk[0, :] <= self.config.MAX_ACCEL]

        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    # def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
    #     """
    #     calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
    #     using the current velocity, calc the T points along the reference path
    #     :param cx: Course X-Position
    #     :param cy: Course y-Position
    #     :param cyaw: Course Heading
    #     :param sp: speed profile
    #     :dl: distance step
    #     :pind: Setpoint Index
    #     :return: reference trajectory ref_traj, reference steering angle
    #     """
    #
    #     # Create placeholder Arrays for the reference trajectory for T steps
    #     ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
    #     ncourse = len(cx)
    #
    #     # Find nearest index/setpoint from where the trajectories are calculated
    #     _, _, _, ind = self.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)
    #
    #     # Load the initial parameters from the setpoint into the trajectory
    #     ref_traj[0, 0] = cx[ind]
    #     ref_traj[1, 0] = cy[ind]
    #     ref_traj[2, 0] = sp[ind]
    #     ref_traj[3, 0] = cyaw[ind]
    #
    #     # based on current velocity, distance traveled on the ref line between time steps
    #     travel = abs(state.v) * self.config.DTK
    #     dind = travel / self.config.dlk
    #     ind_list = int(ind) + np.insert(
    #         np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
    #     ).astype(int)
    #     ind_list[ind_list >= ncourse] -= ncourse
    #     ref_traj[0, :] = cx[ind_list]
    #     ref_traj[1, :] = cy[ind_list]
    #     ref_traj[2, :] = sp[ind_list]
    #     cyaw[cyaw - state.yaw > 4.5] = np.abs(
    #         cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
    #     )
    #     cyaw[cyaw - state.yaw < -4.5] = np.abs(
    #         cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
    #     )
    #     ref_traj[3, :] = cyaw[ind_list]
    #
    #     return ref_traj

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, dl, pind):
        xref = np.zeros((self.config.NXK, self.config.TK + 1))
        dref = np.zeros((1, self.config.TK + 1))
        ncourse = len(cx)
        tref = cyaw[pind]

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0
        travel = 0.0

        if (abs(state.yaw - xref[3, 0]) > 3.14):
            if (state.yaw < xref[3, 0]):
                state.yaw += 2 * math.pi
            else:
                print("Hey you out there in the cold")
                xref[3, 0] += 2 * math.pi

        for i in range(1, self.config.TK + 1):
            travel += abs(state.v) * self.config.DTK
            dind = int(round(travel / dl))
            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0
            if (i > 0):
                if xref[3, i] < 1.0 and xref[3, i - 1] > 6.0:
                    xref[3, i] += 2 * math.pi

        for i in range(self.config.TK - 1, -1, -1):
            if (xref[3, i] - xref[3, i + 1]) < -math.pi:
                xref[3, i] += 2 * math.pi
        if (state.yaw - xref[3, 0]) < -math.pi:
            state.yaw += 2 * math.pi

        return xref, ind, dref

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

    def mpc_prob_solve(self, ref_traj, path_predict, x0, dref):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], dref[0,t]
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        print(A_block.data.shape)
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

    def iterative_linear_mpc_control(self, ref_path, x0, dref, oa, od):
        """
        MPC control with updating operational point iteratively
        """
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK
        for i in range(self.config.MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, ref_path)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.mpc_prob_solve(ref_path, xbar, x0, dref)
            print(oa)
            print(poa)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.config.DU_TH:
                break
        else:
            print("Iterative is max iter")
        return oa, od, ox, oy, oyaw, ov
    # def linear_mpc_control(self, ref_path, x0, oa, od):
    #     """
    #     MPC control with updating operational point iteraitvely
    #     :param ref_path: reference trajectory in T steps
    #     :param x0: initial state vector
    #     :param oa: acceleration of T steps of last time
    #     :param od: delta of T steps of last time
    #     """
    #
    #     # Call the Motion Prediction function: Predict the vehicle motion for x-steps
    #     path_predict = self.predict_motion(x0, oa, od, ref_path)
    #     poa, pod = oa[:], od[:]
    #
    #     # Run the MPC optimization: Create and solve the optimization problem
    #     mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
    #         ref_path, path_predict, x0
    #     )
    #
    #     return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    #
    # def linear_mpc_control(self, xref, xbar, x0, dref):
    #     """
    #     Linear MPC control
    #     xref: reference point
    #     xbar: operational point
    #     x0: initial state
    #     dref: reference steer angle
    #     """
    #     x = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
    #     u = cvxpy.Variable((self.config.NU, self.config.TK))
    #     cost = 0.0
    #     constraints = []
    #     for t in range(self.config.TK):
    #         cost += cvxpy.quad_form(u[:, t], self.config.Rk)
    #         if t != 0:
    #             cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.config.Qk)
    #         A, B, C = self.get_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
    #         constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
    #
    #         if t < (self.config.TK - 1):
    #             cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.config.Rdk)
    #             constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.config.MAX_DSTEER * self.config.DTK]
    #
    #     cost += cvxpy.quad_form(xref[:, self.config.TK] - x[:, self.config.TK], self.config.Qfk)
    #     constraints += [x[:, 0] == x0]
    #     constraints += [x[2, :] <= self.config.MAX_SPEED]
    #     constraints += [x[2, :] >= self.config.MIN_SPEED]
    #     constraints += [cvxpy.abs(u[0, :]) <= self.config.MAX_ACCEL]
    #     constraints += [cvxpy.abs(u[1, :]) <= self.config.MAX_STEER]
    #     prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    #     prob.solve(solver=cvxpy.OSQP, verbose=False)
    #
    #     if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
    #         ox = self.get_nparray_from_matrix(x.value[0, :])
    #         oy = self.get_nparray_from_matrix(x.value[1, :])
    #         ov = self.get_nparray_from_matrix(x.value[2, :])
    #         oyaw = self.get_nparray_from_matrix(x.value[3, :])
    #         oa = self.get_nparray_from_matrix(u.value[0, :])
    #         odelta = self.get_nparray_from_matrix(u.value[1, :])
    #     else:
    #         print("Error: Cannot solve mpc..")
    #         oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
    #     return oa, odelta, ox, oy, oyaw, ov

    def publish_ref_traj(self, pose_msg, ref_traj):
        """
        ref_traj: (x, y, v, yaw) in body frame
        """
        target = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        target.header.frame_id = 'map'
        target.color.r = 0.0
        target.color.g = 0.0
        target.color.b = 1.0
        target.color.a = 1.0
        target.id = 1
        for i in range(ref_traj.shape[1]):
            x, y, yaw, v = ref_traj[:, i]
            # x, y = body2world(pose_msg, x, y)
            print(f'Publishing ref traj x={x}, y={y}')
            print(f'Publishing ref traj yaw={yaw}, v={v}')
            target.points.append(Point(x=x, y=y, z=0.0))
        self.target_pub.publish(target)

    def publish_waypoints(self):
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
        self.waypoint_pub.publish(marker_array)

    def nearest_point(self, point, trajectory):
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
        diffs = trajectory[1:, :] - trajectory[:-1, :]
        l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        dots = np.empty((trajectory.shape[0] - 1,))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t < 0.0] = 0.0
        t[t > 1.0] = 1.0
        projections = trajectory[:-1, :] + (t * diffs.T).T
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp * temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        dx = [state.x - icx for icx in cx[pind:(pind + self.config.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.config.N_IND_SEARCH)]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)
        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))

        if angle < 0:
            mind *= -1
        return ind, mind

    def pi_2_pi(self, angle):
        while (angle > math.pi):
            angle = angle - 2.0 * math.pi
        while (angle < -math.pi):
            angle = angle + 2.0 * math.pi
        return angle

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()


def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
