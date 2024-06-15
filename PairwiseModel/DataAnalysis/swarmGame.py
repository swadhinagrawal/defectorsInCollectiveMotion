import numpy as np
import copy as cp

class Agent:
    def __init__(self,P,id,yaw,pose):
        self.id = id
        self.obj_type = P.getAtt('Follower_type')

        self.yaw = yaw
        self.pose = pose
        self.velocity = np.array([np.cos(self.yaw), np.sin(self.yaw)])*P.getAtt('speed')
        self.rotationAngle = 0.0
        self.temp_yaw = self.yaw
        self.temp_pose = self.pose
        self.temp_velocity = self.velocity
        self.desired_yaw = self.yaw
        # self.strategy = np.random.choice([0,1],p=[0.1,0.9]) # type: ignore
        # if np.linalg.norm(self.id - np.array([int((P.getAtt('nAgents')-1)/2), int((P.getAtt('nAgents')-1)/2)])) == 0.0:#4900:
        # if self.id <= 5 :#4900:
        #     self.strategy = 0
        # else:
        #     self.strategy = 1
        self.strategy = 1
        self.payoff = 0.0
        self.neighbours = self.id
        self.body = None
        self.focal = False
        self.newStrategy = self.strategy
        self.energy = 0.0
        
    def revive(self):
        self.payoff = 0.0
        self.energy = 0.0
        self.focal = False
        self.neighbours = self.id
        self.desired_yaw = self.yaw
        self.temp_yaw = self.yaw
        self.temp_pose = self.pose
        self.temp_velocity = self.velocity
        self.newStrategy = self.strategy
        self.rotationAngle = 0.0
        
    def getAttr(self,name):
        return getattr(self,name)
    
    def setAttr(self,name,val):
        setattr(self,name,val)

# Swarm Models
class PairwiseModel:
    def __init__(self,P):
        print('\nUsing Pairwise interaction model . . .')
        self.agents = [] 
        self.P = P
    
    def createAgents(self):
        for j in range(self.P.getAtt('nAgents')):
            self.agents.append(Agent(P = self.P, id = j, yaw = np.random.uniform(0,2*np.pi), pose = np.random.uniform(0.0 ,self.P.getAtt('home_r') , size = 2)))
    
    def neighbours(self,agents):
        if self.P.getAtt('nAgents')%2==0:
            neighbours = np.random.choice(agents,size = (int(len(agents)/2),2),replace = False)
            for k in neighbours:
                k[0].setAttr('neighbours', k[1].getAttr('id'))
                k[1].setAttr('neighbours', k[0].getAttr('id'))
                focal_agent = np.random.randint(0,2)
                k[focal_agent].setAttr('focal', True)
        else:
            print('Give even number of followers')
    
    def requiredYaw(self,thisAgent,agents):
        new_yaw = agents[thisAgent.neighbours].getAttr('velocity')
        vel_norm = np.linalg.norm(new_yaw)
        if vel_norm!=0:
            new_yaw = new_yaw/vel_norm
        return new_yaw
    
    def clipper(self,val,lim):
        if abs(val)>lim:
            val = lim*(val/abs(val))
        return val
    
    def angleWrapper(self,angle,min_ang,max_ang):
        while angle < min_ang:
            angle += 2*np.pi
        while angle > max_ang:
            angle -= 2*np.pi
        return angle
     
    def headingEstimates(self,thisAgent):
        req_yaw = self.requiredYaw(thisAgent,self.agents)
        
        # if thisAgent.getAttr('strategy') == 1 and self.agents[thisAgent.neighbours].getAttr('strategy') == 1:
        #     steer = np.sum([thisAgent.getAttr('velocity'),req_yaw],axis=0)/np.linalg.norm(thisAgent.getAttr('velocity'))
        # else:
        steer = req_yaw
            
        steer_norm = np.linalg.norm(steer)
        if steer_norm!=0:
            steer = steer/steer_norm
        
        thisAgent.setAttr('desired_yaw', steer)

        turningSide = np.cross(thisAgent.getAttr('velocity'),thisAgent.getAttr('desired_yaw'))
        if turningSide!=0:
            turningSide = turningSide/abs(turningSide)
        elif turningSide == 0:
            turningSide = np.random.choice([-1,1]) # type: ignore
            
        delta_theta = turningSide*np.arccos(self.clipper(np.dot(np.array(thisAgent.getAttr('desired_yaw')),thisAgent.getAttr('velocity')/np.linalg.norm(thisAgent.getAttr('velocity'))),1))
        
        if thisAgent.getAttr('strategy') and thisAgent.getAttr('focal'):    
            thisAgent.setAttr('rotationAngle', self.clipper(delta_theta, self.P.getAtt('omega_max')*self.P.getAtt('dt')))
        else:
            thisAgent.setAttr('rotationAngle', self.clipper(0.0, self.P.getAtt('omega_max')*self.P.getAtt('dt')))
        # thisAgent.setAttr('rotationAngle', self.clipper(delta_theta/self.P.getAtt('dt'), self.P.getAtt('omega_max'))*self.P.getAtt('dt'))

        thisAgent.setAttr('temp_velocity', np.array([[np.cos(thisAgent.getAttr('rotationAngle')), -1*np.sin(thisAgent.getAttr('rotationAngle'))],[np.sin(thisAgent.getAttr('rotationAngle')),np.cos(thisAgent.getAttr('rotationAngle'))]]) @ thisAgent.getAttr('velocity'))
        
        thisAgent.setAttr('temp_yaw', thisAgent.getAttr('yaw') + thisAgent.getAttr('rotationAngle'))
        thisAgent.setAttr('temp_pose', thisAgent.getAttr('pose') + self.P.getAtt('dt')*thisAgent.getAttr('temp_velocity'))
        
    
    # def newPose(self,thisAgent):
    #     thisAgent.setAttr('temp_pose', thisAgent.getAttr('pose') + self.P.getAtt('dt')*thisAgent.getAttr('velocity'))

    def stateUpdate(self,thisAgent):
        thisAgent.setAttr('yaw', thisAgent.getAttr('temp_yaw'))
        thisAgent.setAttr('velocity', thisAgent.getAttr('temp_velocity'))
        thisAgent.setAttr('pose', thisAgent.getAttr('temp_pose'))
        thisAgent.setAttr('energy', -1*(thisAgent.getAttr('rotationAngle')/self.P.getAtt('dt'))**2)
    
    def getAttr(self,name):
        return getattr(self,name)
    
    def setAttr(self,name,val):
        setattr(self,name,val)

# Evolutionary games
class PrisonerDilema:
    def __init__(self,P) -> None:
        print('\nPlaying Prisoner Dilema . . .')
        self.P = P
    
    def payoff(self, thisAgent, thatAgent):
        if thisAgent.getAttr('strategy') == 0 and thatAgent.getAttr('strategy') == 0:
            return self.P.getAtt('PD_P')
        elif thisAgent.getAttr('strategy') == 0 and thatAgent.getAttr('strategy') == 1:
            return self.P.getAtt('PD_T')
        elif thisAgent.getAttr('strategy') == 1 and thatAgent.getAttr('strategy') == 0:
            return self.P.getAtt('PD_S')
        elif thisAgent.getAttr('strategy') == 1 and thatAgent.getAttr('strategy') == 1:
            return self.P.getAtt('PD_R')

    def getNewStrategy(self,agent_A,agent_B):
        # pay_off_diff = np.float128((agent_B.getAttr('payoff')-agent_A.getAttr('payoff')))#abs(np.float128((thatAgent.payoff-thisAgent.payoff)))

        # fermi_fn = 1/(1+np.exp(self.P.getAtt('fermiK') - self.P.getAtt('fermiBeta')*pay_off_diff))

        switch = np.random.uniform(0.0,1.000001)#np.random.choice([0,1], p = [self.P.getAtt('switchProb'),1.0 - self.P.getAtt('switchProb')])#[1.0-fermi_fn,fermi_fn]) # type: ignore
        strategy = cp.copy(agent_A.getAttr('strategy'))
        if switch<=self.P.getAtt('switchProb'):
            strategy = 0
        else:
            strategy = 1
            # strategy = cp.copy(agent_B.getAttr('strategy'))
        # strategy = cp.copy(agent_A.getAttr('strategy'))
        # if agent_B.getAttr('payoff')>agent_A.getAttr('payoff'):
        #     strategy = cp.copy(agent_B.getAttr('strategy'))
        return strategy
    
    # def getNewStrategy(self,agent_A,agent_B):
    #     switch = np.random.uniform(0.0,1.000001)
    #     strategy = cp.copy(agent_A.getAttr('strategy'))
    #     if switch<=self.P.getAtt('switchProb'):
    #         strategy = 0
    #     else:
    #         strategy = 1

    #     return strategy
    
    def update_strategy(self,a):
        a.setAttr('strategy', a.getAttr('newStrategy'))
        
    def game(self,thisAgent, agents):
        # Play the Game here!
        thatAgent = agents[thisAgent.getAttr('neighbours')]
        thisAgent.setAttr('payoff', self.payoff(thisAgent,thatAgent))
        thatAgent.setAttr('payoff', self.payoff(thatAgent,thisAgent))
        thisAgent.setAttr('newStrategy', self.getNewStrategy(thisAgent,thatAgent))
        thatAgent.setAttr('newStrategy', self.getNewStrategy(thatAgent,thisAgent))
        
class Simulator:
    def __init__(self,P):
        # Initialize parameters for simulations
        self.P = P
        # Initialize swarm model
        self.interact = PairwiseModel(self.P)
        self.game = PrisonerDilema(self.P)
        # Set-up follower agents
        self.interact.createAgents()

    def check_reached(self):
        done = False
        if self.P.getAtt('timer')>self.P.getAtt('end_time'):
            done = True
        return done

    def step(self):
        self.interact.neighbours(self.interact.agents)
        
        for a in self.interact.agents:
            # if a.getAttr('strategy') and a.getAttr('focal'):
            self.interact.headingEstimates(a)
            # else:
            #     self.interact.newPose(a)
        
        for a in self.interact.agents:
            # if a.getAttr('strategy'):
            self.interact.stateUpdate(a)

        for a in self.interact.agents:#.flatten(): # type: ignore
            # if a.getAttr('focal'):
            self.game.game(a, self.interact.getAttr('agents'))
        
        for a in self.interact.agents:#.flatten(): # type: ignore
            self.game.update_strategy(a)

 