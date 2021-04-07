import numpy as np 
from matplotlib import pyplot as plt
from numba import njit
import LibFunctions as lib


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
        self.lookahead = 0.5

        self.wpts = None
        self.vs = None
        self.N = None

    def set_wpts(self, wpts, v):
        self.vs = np.ones(len(wpts)) * v
        self.wpts = wpts
        # self.wpts = np.concatenate([wpts, vs[:, None]], axis=-1)
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
        # waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)

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

 
class Oracle(PurePursuit):
    def __init__(self, sim_conf):
        PurePursuit.__init__(self, sim_conf)
        self.name = "Oracle Pure Pursuit"
        
        self.buffer = []

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
            path = path_finder.run_search(0.2)
        except AssertionError:
            return False
        
        # path = find_path(check_fcn, start, end)

        # wpts = optimise_path(path, check_fcn)

        env_map.wpts = path


        # self.pp.set_wpts(wpts)
        self.set_wpts(path, 1)

        return True


    def act(self, obs):
        action = self.act_pp(obs)

        return action

    def done_entry(self, s_prime):
        mem_entry = [s_prime]
        self.buffer.append(mem_entry)


