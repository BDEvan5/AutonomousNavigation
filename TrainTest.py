import numpy as np
from matplotlib import pyplot as plt



def train_vehicle(env, vehicle, steps: int):
    done = False
    state = env.reset()

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            env.render(wait=False)
            # env.render(wait=True)

            vehicle.reset_lap()
            state = env.reset()

    vehicle.t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")


"""General test function"""
def test_single_vehicle(env, vehicle, show=False, laps=100):
    crashes = 0
    completes = 0
    lap_times = [] 

    state = env.reset()
    done, score = False, 0.0
    for i in range(laps):
        print(f"Running lap: {i}")
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
            # env.render(False)
        print(f"Lap time updates: {env.steps}")
        if show:
            # vehicle.show_vehicle_history()
            env.history.show_history()
            # env.history.show_forces()
            env.render(wait=False)
            # env.render(wait=True)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        state = env.reset()
        
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")


def run_oracle(env, vehicle, show=False, steps=100):
    done = False
    state = env.reset()
    vehicle.plan(env.env_map)
    while not vehicle.plan(env.env_map):
        state = env.reset() 

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        
        # env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)
            env.render(wait=False)

            env.history.show_history()

            if s_prime[-1] == -1:
                env.render(wait=True)

            state = env.reset()
            while not vehicle.plan(env.env_map):
                state = env.reset()

    print(f"Finished Training: {vehicle.name}")


def generate_oracle_data(env, vehicle, show=False, steps=100):
    done = False
    state = env.reset()
    vehicle.plan(env.env_map)
    while not vehicle.plan(env.env_map):
        state = env.reset() 

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        
        # env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)

            env.render(wait=False)

            # if s_prime[-1] == -1:
            #     env.render(wait=True)

            state = env.reset()
            while not vehicle.plan(env.env_map):
                state = env.reset()

        if n % 200 == 1:
            print(f"Filling buffer: {n}")

    vehicle.save_buffer("ImitationData1")

    print(f"Finished Training: {vehicle.name}")

    return vehicle.buffer

