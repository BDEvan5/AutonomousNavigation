import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib


# keep in mind that this is used in the mod network
#consider rather keeping a copy to be used in mod network
class PurePursuit:
    def __init__(self, map_name, sim_conf, pp_conf=None) -> None:
        if pp_conf is None:
            pp_conf = lib.load_conf("mod_conf")

        self.name = "Pure Pursuit Path Follower"
        self.path_name = None
        self.map_name = map_name

        mu = sim_conf.mu
        g = sim_conf.g
        self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.f_max = mu * self.m * g #* safety_f

        self.v_gain = pp_conf.v_gain
        self.lookahead = pp_conf.lookahead

        self.wpts = None
        self.vs = None
        self.N = None
        self.ss = None #TODO: I think this isn't needed and can be removed

        self.aim_pts = []

        try:
            # raise FileNotFoundError
            self._load_csv_track()
        except FileNotFoundError:
            print(f"Problem Loading map - generate map pts")

    def _load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        o_ss = self.ss
        o_vs = self.vs
        new_line = []
        new_ss = []
        new_vs = []
        for i in range(self.N-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                ds = o_ss[i+1] - o_ss[i]
                new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.wpts = np.array(new_line)
        self.ss = np.array(new_ss)
        self.vs = np.array(new_vs)
        self.N = len(new_line)

    def _get_current_waypoint(self, position, theta):
        # nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.wpts)
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        if nearest_dist < self.lookahead:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
            if i2 == None:
                return None
            i = i2
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = self.wpts[i2]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < 20:
            return np.append(self.wpts[i], self.vs[i])

    def act(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos, pose_th)

        if lookahead_point is None:
            return 4.0, 0.0

        self.aim_pts.append(lookahead_point)

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.v_gain * speed

        # print(f"Speed: {speed} --> Steer: {steering_angle}")
        # avg_speed = max(speed, obs[3])
        # steering_angle = self.limit_inputs(avg_speed, steering_angle)

        return [steering_angle, speed]

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        # if max_steer < abs(steering_angle):
        #     print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer

    def reset_lap(self):
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)
        
        # print(f"Wpt_y: {waypoint_y} --> pos: {position} --> lookahead: {lookahead_point}")

        speed = lookahead_point[2]
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return speed, steering_angle

    def nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    def show_vehicle_history(self):
        plt.figure(5)
        pts = np.array(self.aim_pts)
        plt.plot(pts[:, 0], pts[:, 1])
        plt.pause(0.001)


# @njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]





