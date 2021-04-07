from Oracle import Oracle
from TrainTest import run_oracle, train_vehicle, test_single_vehicle
from Simulator import NavSim
from AgentNav import AgentNav


def train_nav_std():
    env = NavSim("pfeiffer")
    vehicle = AgentNav("std_nav_test", env.sim_conf)

    train_vehicle(env, vehicle, 50000)

def test_nav_std():

    env = NavSim("pfeiffer")
    vehicle = AgentNav("std_nav_test", env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100)


def test_oracle():
    env = NavSim("pfeiffer")
    vehicle = Oracle(env.sim_conf)

    run_oracle(env, vehicle, True, 1000)


if __name__ == "__main__":
    # test_nav_std()
    # train_nav_std()

    test_oracle()


