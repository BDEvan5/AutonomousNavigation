import numpy as np 
from matplotlib import pyplot as plt
import os

import LibFunctions as lib
from SimUtils import CarModel, ScanSimulator, NavMap, SimHistory

class BaseSim:
    """
    Base simulator class

    Important parameters:
        timestep: how long the simulation steps for
        max_steps: the maximum amount of steps the sim can take

    Data members:
        car: a model of a car with the ability to update the dynamics
        scan_sim: a simulator for a laser scanner
        action: the current action which has been given
        history: a data logger for the history
    """
    def __init__(self, env_map, sim_conf):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.env_map = env_map
        self.sim_conf = sim_conf #TODO: don't store the conf file, just use and throw away.

        self.timestep = self.sim_conf.time_step
        self.max_steps = self.sim_conf.max_steps
        self.plan_steps = self.sim_conf.plan_steps

        self.car = CarModel(self.sim_conf)
        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.set_check_fcn(self.env_map.check_scan_location)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.history = SimHistory(self.sim_conf)
        self.done_reason = ""

    def step_control(self, action):
        """
        Steps the simulator for a single step

        Args:
            action: [steer, speed]
        """
        d_ref = action[0]
        v_ref = action[1]
        acceleration, steer_dot = self.control_system(v_ref, d_ref)
        self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
        self.steps += 1

        return self.done_fcn()

    def step_plan(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """

        for _ in range(self.plan_steps):
            if self.step_control(action):
                break

        self.record_history(action)

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def record_history(self, action):
        self.action = action
        self.history.velocities.append(self.car.velocity)
        self.history.steering.append(self.car.steering)
        self.history.positions.append([self.car.x, self.car.y])
        self.history.thetas.append(self.car.theta)

    def control_system(self, v_ref, d_ref):
        """
        Generates acceleration and steering velocity commands to follow a reference
        Note: the controller gains are hand tuned in the fcn

        Args:
            v_ref: the reference velocity to be followed
            d_ref: reference steering to be followed

        Returns:
            a: acceleration
            d_dot: the change in delta = steering velocity
        """

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def reset(self):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0
        self.reward = 0

        #TODO: move this reset to inside car
        self.env_map.end_goal = self.env_map.generate_location()
        start_pose = self.env_map.generate_location()
        orientation = (np.random.random() - 0.5) * 2* np.pi
        self.env_map.start_pose = np.append(start_pose, orientation)
        self.car.reset_state(self.env_map.start_pose)


        self.history.reset_history()

        return self.get_observation()

    def render(self, wait=False):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column([self.car.x, self.car.y])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def min_render(self, wait=False):
        """
        TODO: deprecate
        """
        fig = plt.figure(4)
        plt.clf()  

        ret_map = self.env_map.scan_map
        plt.imshow(ret_map)

        # plt.xlim([0, self.env_map.width])
        # plt.ylim([0, self.env_map.height])

        s_x, s_y = self.env_map.convert_to_plot(self.env_map.start)
        plt.plot(s_x, s_y, '*', markersize=12)

        c_x, c_y = self.env_map.convert_to_plot([self.car.x, self.car.y])
        plt.plot(c_x, c_y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            r_x, r_y = self.env_map.convert_to_plot(range_val)
            x = [c_x, r_x]
            y = [c_y, r_y]

            plt.plot(x, y)

        for pos in self.action_memory:
            p_x, p_y = self.env_map.convert_to_plot(pos)
            plt.plot(p_x, p_y, 'x', markersize=6)

        text_start = self.env_map.width + 10
        spacing = int(self.env_map.height / 10)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_start, spacing*1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_start, spacing*2, s) 
        s = f"Done: {self.done}"
        plt.text(text_start, spacing*3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_start, spacing*4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_start, spacing*5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_start, spacing*6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_start, spacing*7, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(text_start, spacing*8, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, spacing*9, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
  
    def get_target_obs(self):
        target = self.env_map.end_goal
        pos = [self.car.x, self.car.y]
        angle = lib.get_bearing(pos, target) + self.car.theta
        distance = lib.get_distance(pos, target)

        return [angle, distance]

    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        car_obs = self.car.get_car_state()
        pose = car_obs[0:3]
        scan = self.scan_sim.get_scan(pose)

        target = self.get_target_obs()

        observation = np.concatenate([car_obs, target, scan, [self.reward]])
        return observation

    def done_fcn(self):
        distance_threshold = 0.5
        pos = [self.car.x, self.car.y]
        if lib.get_distance(pos, self.env_map.end_goal) < distance_threshold:
            self.done = True
        elif self.steps > self.max_steps:
            self.done = True
        elif self.env_map.check_scan_location(pos):
            self.done = True

        return self.done


class NavSim(BaseSim):
    def __init__(self, map_name, sim_conf=None):
        """
        Init function

        Args:
            map_name: name of forest map to use.
            sim_conf: config file for simulation
        """
        if sim_conf is None:
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

        env_map = NavMap(map_name)
        if sim_conf is None:
            sim_conf = lib.load_conf("config", "std_config")
        BaseSim.__init__(self, env_map, sim_conf)


        


