from Imitation import BufferIL, ImitationNet, ImitationVehicle
from Oracle import Oracle
from TrainTest import generate_oracle_data, run_oracle, train_vehicle, test_single_vehicle
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

def test_data_generation():
    env = NavSim("pfeiffer")
    vehicle = Oracle(env.sim_conf)

    buffer = generate_oracle_data(env, vehicle, True, 1000)



def test_imitation_training():
    buffer = BufferIL()
    buffer.load_data("ImitationData1")
    
    agent_name = "ImitationPfeiffer"
    imitation_vehicle = ImitationNet(agent_name)
    imitation_vehicle.train(buffer)
    imitation_vehicle.save("Vehicles/")

def test_trained_imitation():
    agent_name = "ImitationPfeiffer"
    
    env = NavSim("pfeiffer")
    vehicle = ImitationVehicle(env.sim_conf, agent_name)
    test_single_vehicle(env, vehicle, True, 100)


if __name__ == "__main__":
    # test_nav_std()
    # train_nav_std()
    # test_oracle()

    # test_data_generation()
    # test_imitation_training()
    test_trained_imitation()


