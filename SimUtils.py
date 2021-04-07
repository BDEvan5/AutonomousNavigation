import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib
from io import FileIO
import numpy as np 
import yaml
from PIL import Image
from scipy import ndimage

import LibFunctions as lib

class CarModel:
    """
    A simple class which holds the state of a car and can update the dynamics based on the bicycle model

    Data Members:
        x: x location of vehicle on map
        y: y location of vehicle on map
        theta: orientation of vehicle
        velocity: 
        steering: delta steering angle
        th_dot: the change in orientation due to steering

    """
    def __init__(self, sim_conf):
        """
        Init function

        Args:
            sim_conf: a config namespace with relevant car parameters
        """
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0
        self.th_dot = 0

        self.prev_loc = 0

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, a, d_dot, dt):
        """
        Updates the internal state of the vehicle according to the kinematic equations for a bicycle model

        Args:
            a: acceleration
            d_dot: rate of change of steering angle
            dt: timestep in seconds

        """
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.th_dot = theta_dot
        dth = theta_dot * dt
        self.theta = lib.add_angles_complex(self.theta, dth)

        a = np.clip(a, -self.max_a, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

    def get_car_state(self):
        """
        Returns the state of the vehicle as an array

        Returns:
            state: [x, y, theta, velocity, steering]

        """
        state = []
        state.append(self.x) #0
        state.append(self.y)
        state.append(self.theta) # 2
        state.append(self.velocity) #3
        state.append(self.steering)  #4

        state = np.array(state)

        return state

    def reset_state(self, start_pose):
        """
        Resets the state of the vehicle

        Args:
            start_pose: the starting, [x, y, theta] to reset to
        """
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]
        self.velocity = 0
        self.steering = 0
        self.prev_loc = [self.x, self.y]


class ScanSimulator:
    """
    A simulation class for a lidar scanner

    Parameters:
        number of beams: number of laser scans to return
        fov: field of view
        std_noise: the standard deviation of the noise which is added to the beams.

    Data members:
        scan_output: the last scan which was returned

    External Functions:
        set_check_fcn(fcn): give a function which can be called to check if a certain location falls in the driveable area
        get_scan(pose): returns a scan

    TODO: njit functions, precompute sines and cosines, improve the step searching

    """
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise
        self.rng = np.random.default_rng(seed=12345)

        self.dth = self.fov / (self.number_of_beams -1)
        self.scan_output = np.zeros(number_of_beams)

        self.step_size = 0.2
        self.n_searches = 20

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, pose):
        """
        A simple function to get a laser scan reading for a given pose.
        Adds noise with a std deviation as in the config file

        Args:
            pose: [x, y, theta] of the vehicle at present state
        
        Returns:
            scan: array of the output from the laser scan.
        """
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i - self.fov/2
            self.scan_output[i] = self._trace_ray(x, y, scan_theta)

        # noise = self.rng.normal(0., self.std_noise, size=self.number_of_beams)
        # self.scan_output = self.scan_output + noise

        return self.scan_output

    def _trace_ray(self, x, y, theta):
        """
        returns the % of the max range finder range which is in the driveable area for a single ray

        Args:
            x: x location
            y: y location
            theta: angle of orientation

        TODO: use pre computed sins and cosines
        """
        # obs_res = 10
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * (j + 1)  # search from 1 step away from the point
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        ray = (j) / self.n_searches #* (1 + np.random.normal(0, self.std_noise))
        return ray

    def set_check_fcn(self, check_fcn):
        """
        Sets the function which is used interally to see if a location is driveable

        Args: 
            check_fcn: a function which can be called with a location as an argument
        """
        self._check_location = check_fcn

#TODO: move this to another location
class SimHistory:
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

    def show_forces(self):
        mu = self.sim_conf['car']['mu']
        m = self.sim_conf['car']['m']
        g = self.sim_conf['car']['g']
        l_f = self.sim_conf['car']['l_f']
        l_r = self.sim_conf['car']['l_r']
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.sim_conf['sim']['timestep'] * self.sim_conf['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)


class NavMap:
    def __init__(self, map_name):
        self.map_name = map_name 

        # map info
        self.resolution = None
        self.map_height = None
        self.map_width = None
        
        self.map_img = None
        self.dt_img = None
        self.end_goal = np.zeros(2)
        self.start_pose = np.zeros(3)

        self.wpts = None

        self.load_map()

    def load_map(self):
        file_name = 'nav_maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            self.resolution = yaml_file['resolution']
            map_img_path = 'nav_maps/' + yaml_file['image']
            
        except Exception as e:
            print(e)
            raise FileIO("Problem loading map yaml file")

        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        self.map_img = self.map_img.T

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        img = np.ones_like(self.map_img) * 255 - self.map_img
        self.dt_img = ndimage.distance_transform_edt(img) * self.resolution
        self.dt_img = np.array(self.dt_img).T

        # plt.figure(1)
        # plt.imshow(self.dt_img, origin='lower')
        # plt.pause(0.001)

        # plt.figure(2)
        # plt.imshow(self.map_img, origin='lower')
        # plt.pause(0.001)

    def generate_location(self):
        obs_threshold = 0.5 # value in meters

        rands = np.random.random(2)
        location = rands * [self.map_width, self.map_height] * self.resolution
        x = int(location[0])
        y = int(location[1])
        i = 0
        while self.dt_img[x, y] < obs_threshold and i < 100:
            rands = np.random.random(2)
            location = rands * [self.map_width, self.map_height] * self.resolution
            x = int(location[0])
            y = int(location[1])
            i += 1

        return location

    def render_map(self, figure_n=1, wait=False):
        #TODO: draw the track boundaries nicely
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        plt.ylim([0, self.map_height])

        plt.imshow(self.map_img, origin='lower')

        x, y = self.xy_to_row_column(self.start_pose)
        plt.plot(x, y, '*', markersize=12, color='g')
        x, y = self.xy_to_row_column(self.end_goal)
        plt.plot(x, y, '*', markersize=12, color='r')

        if self.wpts is not None:
            xs, ys = self.convert_positions(self.wpts)
            plt.plot(xs, ys, 'x', markersize=12)

        plt.pause(0.0001)
        if wait:
            plt.show()
            pass

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resolution, 0, self.map_width+1)))
        r = int(round(np.clip(pt[1] / self.resolution, 0, self.map_height+1)))
        return c, r

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True

        x, y = self.xy_to_row_column(x_in)
        if x >= self.map_width or y >= self.map_height:
            return True
        # if self.map_img[x, y]: # consider using dx???
        if self.dt_img[x, y] == 0: # consider using dx???
            return True

    def check_search_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True

        x, y = self.xy_to_row_column(x_in)
        if x >= self.map_width or y >= self.map_height:
            return True
        if self.dt_img[x, y] < 0.4: # consider using dx???
            return True

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
