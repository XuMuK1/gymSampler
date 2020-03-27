import pythonSimulators as pySims
import gym
import numpy as np
import multiprocessing as mp

#####Test 0: try to sample from StateSampler and check if the dimension is correct
print("Start Test 0: sample from StateSampler")
gymEnv = gym.make("CartPole-v0")
gymEnv.reset()
stateSampler = pySims.GymResetSampler(gymEnv)
print(stateSampler.sample(10))
assert stateSampler.sample(10).shape == (10,4)
print("Test 0 is OK")


#####Test 1: try to reset the state0 from given state0
print("Start Test 1: ResetFromState0")
gymEnv = gym.make("CartPole-v0")
gymEnv.reset()
state0 = np.array([0,1,-0.2,0.5])
sim=pySims.GymSimulator(gymEnv)
sim.ResetFromS0(state0)
eps=1e-17
assert ( np.linalg.norm(state0-sim.GetEnvState())<eps and np.linalg.norm(state0-sim.gymInstance.state)<eps)
print("Test 1 is OK")


#####Test 2: try to reset the state0 from stateSampler and check if the dimensions are correct
print("Start Test 2: ResetFromStateSampler")
gymEnv = gym.make("CartPole-v0")
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
stateSampler = pySims.GymResetSampler(gymEnv)
state0=stateSampler.sample()[0,:]
state01=sim.ResetFromStateSampler(stateSampler)
eps=1e-17
assert state0.shape == sim.GetEnvState().shape
print("Test 2 is OK")


#####Test 3: Check if transitionSampler works correctly for one s0
print("Start Test 3: SampleTransitionFromState0")
gymEnv = gym.make("CartPole-v0")
state0 = np.concatenate([np.array([[0,0.01,-0.01,0.01]]) for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
print('DEBUG TESTS ActionSpaceShape',gymEnv.action_space.shape)
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))
sample = sim.SampleTransitionsFromS0(state0,policy)
print('Test Sample')
print(sample)
assert sample[0].shape == (5,1,4) and sample[1].shape == (5,1,1) and sample[2].shape == (5,1,1)
print("Test 3 is OK")


#####Test 4: Check if transitionSampler works correctly for stateSampler
print("Start Test 4: SampleTransitionFromStateSampler")
gymEnv = gym.make("CartPole-v0")
state0 = np.concatenate([np.array([[0,0.01,-0.01,0.01]]) for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
stateSampler = pySims.GymResetSampler(gymEnv)
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))
sample = sim.SampleTransitionsFromStateSampler(stateSampler,policy,Ns0=5)
print('Test Sample')
print(sample)
assert sample[0].shape == (5,1,4) and sample[1].shape == (5,1,1) and sample[2].shape == (5,1,1)
print("Test 4 is OK")



print("Start Test 5: Sample Trajectories From S0s")
gymEnv = gym.make("CartPole-v0")
print('DEBUG',gymEnv.reset())
state0 = np.concatenate([gymEnv.reset()[None,:] for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))
sample = sim.SampleTrajectoriesFromS0(state0,policy)
print('DEBUG Test Sample')
print('****states')
print(sample[0])
print('****actions')
print(sample[1])
print('****rewards')
print(sample[2])
print('DEBUG: statesShape',sample[0].shape)
print('DEBUG: actionsShape',sample[1].shape)
print('DEBUG: rewardsShape',sample[2].shape)
assert sample[0].shape[0] == 5 and sample[0].shape[1] == 4
assert sample[1].shape[0] == 5 and sample[1].shape[1] == 1
assert sample[2].shape[0] == 5 and sample[2].shape[1] == 1
print("Test 5 is OK")


print("Start Test 6: Sample Trajectories From StateSampler")
gymEnv = gym.make("CartPole-v0")
print('DEBUG',gymEnv.reset())
state0 = np.concatenate([gymEnv.reset()[None,:] for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
stateSampler = pySims.GymResetSampler(gymEnv)
Ns0=5
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))
sample = sim.SampleTrajectoriesFromStateSampler(stateSampler,policy,Ns0=Ns0)
print('DEBUG Test Sample')
print('****states')
print(sample[0])
print('****actions')
print(sample[1])
print('****rewards')
print(sample[2])
print('DEBUG: statesShape',sample[0].shape)
print('DEBUG: actionsShape',sample[1].shape)
print('DEBUG: rewardsShape',sample[2].shape)
assert sample[0].shape[0] == 5 and sample[0].shape[1] == 4
assert sample[1].shape[0] == 5 and sample[1].shape[1] == 1
assert sample[2].shape[0] == 5 and sample[2].shape[1] == 1
print("Test 6 is OK")


print("Start Test 7: Sample Trajectories From S0s (Parallel)")
gymEnv = gym.make("CartPole-v0")
print('DEBUG',gymEnv.reset())
state0 = np.concatenate([gymEnv.reset()[None,:] for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))

pool = mp.Pool(mp.cpu_count()-1)
sample = sim.SampleTrajectoriesFromS0Parallel(pool,state0,policy)
pool.close()
print('DEBUG Test Sample')
print('****states')
print(sample[0])
print('****actions')
print(sample[1])
print('****rewards')
print(sample[2])
print('DEBUG: statesShape',sample[0].shape)
print('DEBUG: actionsShape',sample[1].shape)
print('DEBUG: rewardsShape',sample[2].shape)
assert sample[0].shape[0] == 5 and sample[0].shape[1] == 4
assert sample[1].shape[0] == 5 and sample[1].shape[1] == 1
assert sample[2].shape[0] == 5 and sample[2].shape[1] == 1
print("Test 7 is OK")


print("Start Test 8: Sample Trajectories From StateSampler")
gymEnv = gym.make("CartPole-v0")
print('DEBUG',gymEnv.reset())
state0 = np.concatenate([gymEnv.reset()[None,:] for k in np.arange(5)],axis=0)
gymEnv.reset()
sim=pySims.GymSimulator(gymEnv)
stateSampler = pySims.GymResetSampler(gymEnv)
Ns0=5
def policy(s):
    #returns action based on state
    return np.random.choice(np.arange(sim.actionSpaceN))
pool = mp.Pool(mp.cpu_count()-1)
sample = sim.SampleTrajectoriesFromStateSamplerParallel(pool,stateSampler,policy,Ns0=Ns0)
pool.close()
print('DEBUG Test Sample')
print('****states')
print(sample[0])
print('****actions')
print(sample[1])
print('****rewards')
print(sample[2])
print('DEBUG: statesShape',sample[0].shape)
print('DEBUG: actionsShape',sample[1].shape)
print('DEBUG: rewardsShape',sample[2].shape)
assert sample[0].shape[0] == 5 and sample[0].shape[1] == 4
assert sample[1].shape[0] == 5 and sample[1].shape[1] == 1
assert sample[2].shape[0] == 5 and sample[2].shape[1] == 1
print("Test 8 is OK")
