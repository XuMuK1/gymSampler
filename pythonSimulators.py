import gym
import numpy as np
#import tqdm.tqdm as tqdm
import multiprocessing as mp


class Error(Exception):
    pass
class ShapeMismatchError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class StateSampler:
    #UNDER CONSTRUCTION!!!
    def __init__(self):
        pass

    def sample(self,Nsamples):
        #samples Nsamples
        #returns [NSamples,]
        pass

class GymResetSampler(StateSampler):
    def __init__(self, gymEnv):
        #gymEnv -- gym environment
        #be careful to create separate copy of it
        self.gymEnv = gymEnv
        self.stateSpaceType = 'Continuous'
        self.stateSpaceShape = gymEnv.reset().shape
        if(len(self.stateSpaceShape)==0):
            self.stateSpaceShape = (1,)
            self.stateSpaceN = self.gymEnv.observation_space.n
            self.stateSpaceType='Discrete'


    def sample(self,Nsamples=1):
        print("StateSampler.sample.DEBUG","reset:",self.gymEnv.reset())
        if(self.stateSpaceType=='Discrete'):
            return np.concatenate([np.array([self.gymEnv.reset()])[None,:] for k in np.arange(Nsamples)], axis=0)
        else:
            return np.concatenate([self.gymEnv.reset()[None,:] for k in np.arange(Nsamples)], axis=0)



class GymSimulator:

    #UNDER CONSTRUCTION!!!
    #supports only Box and Discrete action and state spaces!
    def __init__(self,gymInst):
        #gymInstance -- gym environment
        #the rest are its parameters
        self.gymInstance = gymInst

        self.stateSpaceType = 'Continuous'
        self.stateSpaceShape = gymInst.reset().shape
        if(len(self.stateSpaceShape)==0):
            self.stateSpaceShape = (1,)
            self.stateSpaceN = self.gymInstance.observation_space.n
            self.stateSpaceType='Discrete'

        self.actionSpace = gymInst.action_space
        self.actionSpaceShape = self.actionSpace.shape
        self.actionSpaceType='Continuous'
        if(len(self.actionSpaceShape)==0):
            self.actionSpaceShape=(1,)
            self.actionSpaceN=self.actionSpace.n
            self.actionSpaceType='Discrete'


        print("**********GymSimulator is set,")
        print("*******stateSpaceShape",self.stateSpaceShape)
        print("*******actionSpace",self.actionSpace)
        print("*******actionSpaceShape",self.actionSpaceShape)
        print("*******actionSpaceType",self.actionSpaceType)


    def GetEnvState(self):
        return self.gymInstance.state

    ######RESETS the environment to given state
    def ResetFromS0(self, s0):
        self.gymInstance.reset()
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0

    def ResetFromStateSampler(self, stateSampler):
        self.gymInstance.reset()
        s0=stateSampler.sample()[0,:]
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0



    ######Transition Samplers(Sequential and Parallel)
    def SampleTransitionsFromS0(self, s0, policy, NNested=1, returnRewards=False):
        #samples transitions from state s0
        # s0 -- [batch,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        # NNested -- number of samples from each of states in s0
        # returnRewards -- whether the method should return ONLY rewards or not
        #returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], [batch,NNested,1]
        # or (if returnRewards)
        #returns [batch,NNested,1]
    
        #print('DEBUG.SampleTransitionsFromS0')


        if (not len(s0.shape) == 2):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not valid, only [batch,StateSpaceShape] is acceptable")
        if (not s0.shape[1] == self.stateSpaceShape[0]):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested,self.stateSpaceShape[0]])
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,:])
                action = policy(s0[s0Id,:])
                state,reward,done,_ = self.gymInstance.step(action)

                #print('*************state',state)
                #print('*************reward',reward)
                #print('*************done',done)
                #print(1/0)

                if(not returnRewards):
                    states[s0Id,nestedId,:] = state
                    actions[s0Id,nestedId,:] = action
                    dones[s0Id,nestedId,0] = done
                rewards[s0Id,nestedId,:] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards,dones
        return rewards



    def SampleTransitionsFromStateSampler(self, stateSampler, policy, Ns0=1, NNested=1, returnRewards=False):
        #samples transitions from state s0 ~ stateSampler
        # stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        # Ns0 -- number of states in s0, from which it is needed to sample transitions
        # NNested -- number of samples from each of states in s0
        # returnRewards -- whether the method should return ONLY rewards or not
        #returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], 
        #        [batch,NNested,1], [batch,NNested,1]
        # or (if returnRewards)
        #returns [batch,NNested,1]
        s0 = stateSampler.sample(Ns0)
        if (not len(s0.shape) == 2):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not valid, only [batch,StateSpaceShape] is acceptable")
        if (not s0.shape[1] == self.stateSpaceShape[0]):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested,self.stateSpaceShape[0]])
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,:])
                action = policy(s0[s0Id,:])
                state,reward,done,_ = self.gymInstance.step(action)

                if(not returnRewards):
                    states[s0Id,nestedId,:] = state
                    actions[s0Id,nestedId,:] = action
                    dones[s0Id,nestedId,0] = done    
                rewards[s0Id,nestedId,0] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards, dones
        return rewards



    ######TRAJECTORY samplers(Sequential and Parallel), to sample trajectories
    def SampleTrajectoriesFromS0(self, s0, policy, returnRewards=False, maxIterations=400):
        #samples Ntrajs rajectories from given state(s) s0
        # s0 -- [batch,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        # returnRewards -- whether the method should return ONLY rewards or not
        #returns [batch,stateSpaceShape,time],[batch,actionSpaceShape,time-1],[batch,1,time-1]
        # this is states, actions, rewards; 
        #       time is a variable dimension: length of the longest trajectory
        # or (if returnRewards)
        #returns [batch,1,time(list dimension?)]
        #print('DEBUG.simulators.GymSimulator.SampleTrajFromS0')

        if(not returnRewards):
            states =  [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]
        rewards = [0]*s0.shape[0]
        #print('********init states',states)
        maxLen=0
        expandActionDim=False
        #print('********s0',s0)
        #print('********policy',policy(s0[0,:]))
        if(len(policy(s0[0,:]).shape)==0):
            expandActionDim=True
        
        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,:])
            states[trajId] = [self.GetEnvState()]
            actions[trajId] = []
            rewards[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):
                
                #print('************states[trajId][iterId]',states[trajId])
                #print('************states',states)
                action = policy(states[trajId][iterId])
                state,reward,done,_=self.gymInstance.step(action)
                #print('************action',action)
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1
    
            #print('***********done',done)
            if(len(states[trajId])>maxLen):
                maxLen = len(states[trajId])
            #print('***********length',len(states[trajId]))
            
        #wrap np.array around them with zero padding 
        statesNP = np.zeros([s0.shape[0],s0.shape[1],maxLen])
        actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])
        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,:,:len(states[k])] = np.array(states[k]).transpose()        
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        return statesNP,actionsNP,rewardsNP        


    def SampleTrajectoriesFromStateSampler(self, stateSampler, policy, Ns0=1,\
                                             returnRewards=False, maxIterations=400):
        #samples Ntrajs rajectories from given state(s) s0
        # stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        # returnRewards -- whether the method should return ONLY rewards or not
        #returns [Ns0,NNested,stateSpaceShape,time], [Ns0,NNested,actionSpaceShape,time], [Ns0,NNested,1,time]
        # this is states, actions, rewards
        # or (if returnRewards)
        #returns [Ns0,NNested,1,time(list dimension?)]
        
        s0 = stateSampler.sample(Ns0)
        if(not returnRewards):
            states =  [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]
        rewards = [0]*s0.shape[0]
        #print('********init states',states)
        maxLen=0
        expandActionDim=False
        #print('********s0',s0)
        #print('********policy',policy(s0[0,:]))
        if(len(policy(s0[0,:]).shape)==0):
            expandActionDim=True
        
        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,:])
            states[trajId] = [self.GetEnvState()]
            actions[trajId] = []
            rewards[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):
                
                #print('************states[trajId][iterId]',states[trajId])
                #print('************states',states)
                action = policy(states[trajId][iterId])
                state,reward,done,_=self.gymInstance.step(action)
                #print('************action',action)
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1
    
        #samples Ntrajs rajectories from given state(s) s0
            #print('***********done',done)
            if(len(states[trajId])>maxLen):
                maxLen = len(states[trajId])
            #print('***********length',len(states[trajId]))
            
        #wrap np.array around them with zero padding 
        statesNP = np.zeros([s0.shape[0],s0.shape[1],maxLen])
        actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])
        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,:,:len(states[k])] = np.array(states[k]).transpose()        
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        return statesNP,actionsNP,rewardsNP

    
    ######TRAJECTORY samplers(Parallel), to sample trajectories
    def SampleTrajectoriesFromS0Parallel(self, pool,s0, policy, maxIterations=400):
        #samples Ntrajs rajectories from given state(s) s0
        # s0 -- [batch,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        #returns [batch,stateSpaceShape,time],[batch,actionSpaceShape,time-1],[batch,1,time-1]
        # this is states, actions, rewards; 
        #       time is a variable dimension: length of the longest trajectory
        #print('DEBUG.simulators.GymSimulator.SampleTrajFromS0')

        seeds = (1+np.arange(s0.shape[0]))*1035
        trajs = pool.starmap(self.SampleTrajectoriesFromS0ParallelOne, \
                         [(s0[k,:],policy,maxIterations,seeds[k]) for k in np.arange(s0.shape[0])]  )

        #zero padding
        #wrap np.array around them with zero padding 
        #compute maxLen
        maxLen = np.amax([len(trajs[k][0]) for k in np.arange(len(trajs))])
        statesNP = np.zeros([len(trajs),s0.shape[1],maxLen])
        actionsNP = np.zeros([len(trajs),self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([len(trajs),1,maxLen-1])
        for k in np.arange(len(trajs)):
            statesNP[k,:,:len(trajs[k][0])] = np.array(trajs[k][0]).transpose()        
            actionsNP[k,:,:len(trajs[k][1])] = np.array(trajs[k][1]).transpose()
            #print("DEBUG PARALLEL SAMPLING, ZERO PADDING")
            #print("********************",rewardsNP[k,0,:].shape, maxLen, np.array(trajs[k][2]).shape )
            rewardsNP[k,0,:len(trajs[k][2])] = np.array(trajs[k][2])

        return statesNP,actionsNP,rewardsNP                



    def SampleTrajectoriesFromS0ParallelOne(self,s0,policy,maxIterations,seed):
    # handler for parallel sampling (above)
    #   s0 -- [stateSpaceShape]
    #   policy -- a function state --> action, may include sampler from distribution
    #   returns [list(time),stateSpaceShape],[list(time-1),actionSpaceShape],[list(time-1),1]
    #     this is states, actions, rewards; 
    #     time is a variable dimension: length of the longest trajectory

        np.random.seed(seed)
        states =  [] # lists of lists
        actions = []
        rewards = []
        #print('********init states',states)
        maxLen=0
        expandActionDim=False
        #print('********s0',s0)
        #print('********policy',policy(s0[0,:]))
        if(len(policy(s0).shape)==0):
            expandActionDim=True
        
        self.ResetFromS0(s0)
        states.append(self.GetEnvState())

        iterId=0
        done=False
        while(iterId<maxIterations and (not done)):
               
            action = policy(states[iterId])
            state,reward,done,_=self.gymInstance.step(action)
            
            if(expandActionDim):
                action=np.array([action])

            states.append(state)
            actions.append(action)

            rewards.append(reward)
            iterId = iterId + 1
            
        #return statesNP,actionsNP,rewardsNP        
        return states,actions,rewards


    
    def SampleTrajectoriesFromStateSamplerParallel(self, pool,stateSampler, policy, Ns0=1, maxIterations=400):
        #samples Ntrajs rajectories from given stateSampler
        # stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
        # policy -- a function state --> action, may include sampler from distribution
        # returnRewards -- whether the method should return ONLY rewards or not
        #returns [Ns0,NNested,stateSpaceShape,time(list dimension?)], [Ns0,NNested,actionSpaceShape,time(list dimension?)], [Ns0,NNested,1,time(list dimension?)]
        # this is states, actions, rewards
        # or (if returnRewards)
        #returns [Ns0,NNested,1,time(list dimension?)]

        s0 = stateSampler.sample(Ns0)
        seeds = (1+np.arange(s0.shape[0]))*1035 # !!TODO temporary solution, see if you can do this more safely
        trajs = pool.starmap(self.SampleTrajectoriesFromS0ParallelOne, \
                         [(s0[k,:],policy,maxIterations,seeds[k]) for k in np.arange(s0.shape[0])]  )

        #zero padding
        #wrap np.array around them with zero padding 
        #compute maxLen
        maxLen = np.amax([len(trajs[k][0]) for k in np.arange(len(trajs))])
        statesNP = np.zeros([len(trajs),s0.shape[1],maxLen])
        actionsNP = np.zeros([len(trajs),self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([len(trajs),1,maxLen-1])
        for k in np.arange(len(trajs)):
            statesNP[k,:,:len(trajs[k][0])] = np.array(trajs[k][0]).transpose()        
            actionsNP[k,:,:len(trajs[k][1])] = np.array(trajs[k][1]).transpose()
            #print("DEBUG PARALLEL SAMPLING, ZERO PADDING")
            #print("********************",rewardsNP[k,0,:].shape, maxLen, np.array(trajs[k][2]).shape )
            rewardsNP[k,0,:len(trajs[k][2])] = np.array(trajs[k][2])

        return statesNP,actionsNP,rewardsNP                



    
