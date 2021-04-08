import numpy as np 
from matplotlib import pyplot as plt
from numba import njit
import LibFunctions as lib
import casadi as ca
from Imitation import BufferIL


class PathFinder:
    def __init__(self, o_function, start, end):
        # ds is the search size around the current node
        self.ds = None
        self.obstacle_function = o_function

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        self.current_node = Node()
        self.open_node_n = 0

        self.start = start
        self.end = end

    def set_directions(self):
        for i in range(3):
            for j in range(3):
                direction = [(j-1)*self.ds, (i-1)*self.ds]
                self.position_list.append(direction)

        self.position_list.pop(4) # remove stand still

        # this makes it not go diagonal
        self.straight_list = [[self.ds, 0], [0, -self.ds], [-self.ds, 0], [0, self.ds]]
        for pos in self.position_list:
            pos[0] = pos[0] #* self.ds
            pos[1] = pos[1] #* self.ds
        # print(self.position_list)

    def run_search(self, ds, max_steps=1000):
        self.ds = ds
        self.set_directions()
        self.set_up_start_node()
        i = 0
        while len(self.open_list) > 0 and i < max_steps:
            self.find_current_node()

            if self.check_done():
                # print("The shortest path has been found")
                break
            self.generate_children()
            i += 1
            # self.plot_nodes()

            if i % 100 == 0:
                print(f"Search step: {i}")
        # print(f"Number of itterations: {i}")
        assert i < max_steps, "Max Iterations reached: problem with search"
        assert len(self.open_list) > 0, "Search broke: no open nodes"
        path = self.get_path_list()

        return path

    def set_up_start_node(self):
        self.start_n = Node(None, np.array(self.start))
        self.end_n = Node(None, np.array(self.end))

        self.open_list.append(self.start_n)

    def find_current_node(self):
        self.current_node = self.open_list[0]
        current_index = 0
        for index, item in enumerate(self.open_list):
            if item.f < self.current_node.f:
                self.current_node = item
                current_index = index
        # Pop current off open list, add to closed list
        self.open_list.pop(current_index)
        self.closed_list.append(self.current_node)
        # self.logger.debug(self.current_node.log_msg())

    def check_done(self):
        dx_max = self.ds * 1.2
        dis = lib.get_distance(self.current_node.position, self.end_n.position)
        if dis < dx_max:
            # print("Found")
            return True
        return False

    def _check_closed_list(self, new_node):
        for closed_child in self.closed_list:
            if (new_node == closed_child).all():
                return True
        return False

    def _check_open_list(self, new_node):
        for open_node in self.open_list:
            if (new_node == open_node).all(): # if in open set return true
                if new_node.g < open_node.g:
                    open_node.g = new_node.g
                    open_node.parent = new_node.parent
                return True
        return False

    def generate_children(self):
        self.children.clear() # deletes old children

        for direction in self.position_list:
            new_position = lib.add_locations(self.current_node.position, direction)

            # if self.obstacle_function(new_position, self.current_node.position): 
            if self.obstacle_function(new_position): 
                continue 
            new_node = Node(self.current_node, np.array(new_position))

            if self._check_closed_list(new_node): 
                continue
           
            # Create the f, g, and h values
            # takes if it is straight or not into cost consideration.
            if direction in self.straight_list:
                new_node.g = self.current_node.g + self.ds
            else:
                new_node.g = self.current_node.g + self.ds * np.sqrt(2)
            # new_node.g = self.current_node.g + self.ds
            h_val = lib.get_distance(new_node.position, self.end_n.position)
            new_node.h = h_val 
            new_node.f = new_node.g + new_node.h

             # Child is already in the open list
            if self._check_open_list(new_node):
                # if 
                continue

            self.open_list.append(new_node)
            self.open_node_n += 1
            # self.plot_nodes()

    def get_path_list(self):
        curr = self.current_node
        pos_list = []
        while curr is not None:
            pos_list.append(curr.position)
            curr = curr.parent
        pos_list = pos_list[::-1]

        pos_list.append(self.end)

        pos_list = np.array(pos_list)

        return pos_list

    def plot_nodes(self):
        plt.figure(2)
        plt.clf()
        for node in self.closed_list:
            plt.plot(node.position[0], node.position[1], 'x')
        for node in self.open_list:
            plt.plot(node.position[0], node.position[1], '+')

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        plt.pause(0.001)


class Node():
    """A node class for A* Pathfinding"""
    # helper class for a star

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class PurePursuit:
    def __init__(self, sim_conf) -> None:
        self.wpts = None 

        mu = sim_conf.mu
        g = sim_conf.g
        self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.f_max = mu * self.m * g #* safety_f

        self.v_gain = 0.95
        self.lookahead = 1

        self.wpts = None
        self.vs = None
        self.N = None

    def set_wpts(self, wpts, v):
        self.vs = np.ones(len(wpts)) * v
        self.wpts = wpts
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

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

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos, pose_th)

        if lookahead_point is None:
            return 0.0, 4.0

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.v_gain * speed

        return [steering_angle, speed]

    def reset_lap(self):
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)

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


@njit(fastmath=False, cache=True)
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

 
class PathOptimiser:
    def __init__(self, wpts, env_map) -> None:
        self.wpts = wpts
        self.env_map = env_map

        self.nvecs = None
        self.widths = None

    def optimise_path(self):
        self.find_nvecs()
        self.find_widths()

        return self.optimise_trajectory()

    def find_nvecs(self):
        N = len(self.wpts)
        track = self.wpts

        nvecs = []
        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[0, :], track[1, :]))
        nvecs.append(nvec)
        for i in range(1, len(track)-1):
            pt1 = track[i-1]
            pt2 = track[min((i, N)), :]
            pt3 = track[min((i+1, N-1)), :]

            th1 = lib.get_bearing(pt1, pt2)
            th2 = lib.get_bearing(pt2, pt3)
            if th1 == th2:
                th = th1
            else:
                dth = lib.sub_angles_complex(th1, th2) / 2
                th = lib.add_angles_complex(th2, dth)

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
            nvecs.append(nvec)

        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[-2, :], track[-1, :]))
        nvecs.append(nvec)

        self.nvecs = np.array(nvecs)

    def find_widths(self):
        widths = np.zeros_like(self.wpts)
        for i, pt in enumerate(self.wpts):
            x, y = self.env_map.xy_to_row_column(pt)
            #TODO: change this to get individual widths for each side
            w = min(self.env_map.dt_img[x, y] * 0.8, 0.5) 
            widths[i, 0] = w
            widths[i, 1] = w

        self.widths = widths

    def optimise_trajectory(self):
        n_set = MinCurvatureTrajectory(self.wpts, self.nvecs, self.widths)

        d_pts = n_set * self.nvecs
        wpts = self.wpts + d_pts

        return wpts



def MinCurvatureTrajectory(pts, nvecs, ws):
    """
    This function uses optimisation to minimise the curvature of the path
    """
    w_min = - ws[:, 0] * 0.9
    w_max = ws[:, 1] * 0.9
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    N = len(pts)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [pts[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [pts[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    # thetas = np.array(x_opt[1*N:2*(N-1)])

    return n_set




class Oracle(PurePursuit):
    def __init__(self, sim_conf):
        PurePursuit.__init__(self, sim_conf)
        self.name = "Oracle Pure Pursuit"
        
        self.buffer = BufferIL()

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

    def plan(self, env_map):
        """
        Plans a path to be executed by the agent
        Searches for a viable path and then optimises its

        Args:
            map (NavMap): requires a start pose and end goal and 
                a method to check if a location is occupied or not

        """
        start = env_map.start_pose[0:2]
        end = env_map.end_goal
        check_fcn = env_map.check_search_location

        # discritize somehow
        path_finder = PathFinder(check_fcn, start, end)
        try:
            path = path_finder.run_search(0.4)
        except AssertionError:
            return False
        

        # path_optimiser = PathOptimiser(path, env_map)
        # wpts = path_optimiser.optimise_path()
        wpts = path # no optimisation 

        env_map.wpts = wpts
        self.set_wpts(wpts, 1)

        return True


    def act(self, obs):
        action = self.act_pp(obs)

        nn_obs = self.transform_obs(obs)
        nn_act = action[0] / self.max_steer
        self.buffer.add((nn_obs, nn_act))

        return action

    def done_entry(self, s_prime):
        pass

    def transform_obs(self, obs):
        max_angle = np.pi

        # cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        target_angle = [obs[5]/max_angle]
        # target_distance = [obs[6]/self.distance_scale]

        scan = obs[7:-1]

        # nn_obs = np.concatenate([cur_v, cur_d, target_angle, target_distance, scan])
        nn_obs = np.concatenate([cur_d, target_angle, scan])

        return nn_obs

    def save_buffer(self, name):
        buffer = np.array(self.buffer.storage)
        filename = "Vehicles/" + name
        np.save(filename, buffer)


