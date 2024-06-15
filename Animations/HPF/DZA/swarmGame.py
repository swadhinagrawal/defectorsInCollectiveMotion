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
        self.zoa = []
        self.zoo = []
        self.zor = []
        self.Rr = P.getAtt('Rr')
        self.Ro = P.getAtt('Ro')
        self.Ra = P.getAtt('Ra')
        self.fov = P.getAtt('fov')
        self.desired_yaw = self.yaw
        self.neighbours = []
        self.strategy = np.random.choice([0,1],p=[P.getAtt('fracDefectors'),1.0-P.getAtt('fracDefectors')]) # type: ignore
        # if np.linalg.norm(self.id - np.array([int((P.getAtt('nAgents')-1)/2), int((P.getAtt('nAgents')-1)/2)])) == 0.0:#4900:
        # if self.id <= 5 :#4900:
        #     self.strategy = 0
        # else:
        #     self.strategy = 1
        # self.strategy = 0

        self.payoff = 0.0
        self.body = None
        self.newStrategy = self.strategy
        self.energy = 0.0
        
    def revive(self):
        self.payoff = 0.0
        self.zor = []
        self.zoo = []
        self.zoa = []
        self.neighbours = []
        self.desired_yaw = self.yaw
        self.temp_yaw = self.yaw
        self.temp_pose = self.pose
        self.temp_velocity = self.velocity
        self.newStrategy = self.strategy
        self.rotationAngle = 0.0
        self.energy = 0.0
        
    def getAttr(self,name):
        return getattr(self,name)
    
    def setAttr(self,name,val):
        setattr(self,name,val)

# Swarm Model
class CouzinModel:
    def __init__(self,P):
        print('\nUsing Couzin model . . .')
        self.agents = [] 
        self.P = P
    
    def createAgents(self):
        # rng = np.random.default_rng(10)
        # for j in range(self.P.getAtt('nAgents')):
        #     self.agents.append(Agent(P = self.P, id = j, yaw = rng.random()*2*np.pi + np.random.normal(0,np.pi/36), pose = rng.random(size=2)*50))#np.random.uniform(0,2*np.pi)
        for j in range(self.P.getAtt('nAgents')):
            self.agents.append(Agent(P = self.P, id = j, yaw = np.random.uniform(0.0,2.0*np.pi), pose = np.random.uniform(0.0 ,self.P.getAtt('home_r') , size = 2)))#np.random.uniform(0,2*np.pi)
    
    #ALLD 
    def neighbours_all(self, thisAgent, agents):
        for agent in agents:
            d = np.linalg.norm(agent.getAttr('pose') - thisAgent.getAttr('pose'))
            if agent.getAttr('id') != thisAgent.getAttr('id'):
                relative_bearing = self.blindRegion(thisAgent,agent)
                if d <= thisAgent.getAttr('Rr') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zor',thisAgent.getAttr('zor')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Rr') < d <= thisAgent.getAttr('Ro') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zoo',thisAgent.getAttr('zoo')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Ro') < d <= thisAgent.getAttr('Ra') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zoa',thisAgent.getAttr('zoa')+[agent.getAttr('id')])
                else:
                    pass
        thisAgent.setAttr('neighbours',thisAgent.getAttr('zor')+thisAgent.getAttr('zoo')+thisAgent.getAttr('zoa'))
    
    #ZOOD
    def neighbours_zo(self, thisAgent, agents):
        for agent in agents:
            d = np.linalg.norm(agent.getAttr('pose') - thisAgent.getAttr('pose'))
            if agent.getAttr('id') != thisAgent.getAttr('id'):
                relative_bearing = self.blindRegion(thisAgent,agent)
                if d <= thisAgent.getAttr('Rr') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zor',thisAgent.getAttr('zor')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Rr') < d <= thisAgent.getAttr('Ro') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zoo',thisAgent.getAttr('zoo')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Ro') < d <= thisAgent.getAttr('Ra') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zoa',thisAgent.getAttr('zoa')+[agent.getAttr('id')])
                else:
                    pass
        thisAgent.setAttr('neighbours',thisAgent.getAttr('zor')+thisAgent.getAttr('zoo')+thisAgent.getAttr('zoa'))
    
    #ZORD     
    def neighbours_zr(self, thisAgent, agents):
        for agent in agents:
            d = np.linalg.norm(agent.getAttr('pose') - thisAgent.getAttr('pose'))
            if agent.getAttr('id') != thisAgent.getAttr('id'):
                relative_bearing = self.blindRegion(thisAgent,agent)
                if d <= thisAgent.getAttr('Rr') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zor',thisAgent.getAttr('zor')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Rr') < d <= thisAgent.getAttr('Ro') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zoo',thisAgent.getAttr('zoo')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Ro') < d <= thisAgent.getAttr('Ra') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zoa',thisAgent.getAttr('zoa')+[agent.getAttr('id')])
                else:
                    pass
        thisAgent.setAttr('neighbours',thisAgent.getAttr('zor')+thisAgent.getAttr('zoo')+thisAgent.getAttr('zoa'))
            
    #ZOAD
    def neighbours_za(self, thisAgent, agents):
        for agent in agents:
            d = np.linalg.norm(agent.getAttr('pose') - thisAgent.getAttr('pose'))
            if agent.getAttr('id') != thisAgent.getAttr('id'):
                relative_bearing = self.blindRegion(thisAgent,agent)
                if d <= thisAgent.getAttr('Rr') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zor',thisAgent.getAttr('zor')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Rr') < d <= thisAgent.getAttr('Ro') and relative_bearing <= thisAgent.getAttr('fov'):
                    thisAgent.setAttr('zoo',thisAgent.getAttr('zoo')+[agent.getAttr('id')])
                elif thisAgent.getAttr('Ro') < d <= thisAgent.getAttr('Ra') and relative_bearing <= thisAgent.getAttr('fov') and thisAgent.getAttr('strategy'):
                    thisAgent.setAttr('zoa',thisAgent.getAttr('zoa')+[agent.getAttr('id')])
                else:
                    pass
        thisAgent.setAttr('neighbours',thisAgent.getAttr('zor')+thisAgent.getAttr('zoo')+thisAgent.getAttr('zoa'))
    
    
    def blindRegion(self,thisAgent,otherAgent):
        pointing_vec = otherAgent.getAttr('pose') - thisAgent.getAttr('pose')
        norm = np.linalg.norm(pointing_vec)
        vel = np.linalg.norm(thisAgent.getAttr('velocity'))
        pointing_vec = pointing_vec/norm
        if vel!=0:
            dot = np.dot(pointing_vec,thisAgent.getAttr('velocity'))/vel
        else:
            dot = 0
        relative_bearing = np.arccos(np.round(dot,decimals=5))
        return relative_bearing

    def repulsion(self,thisAgent,agents):
        repel = np.zeros(2)
        for i in thisAgent.getAttr('zor'):
            vec = agents[i].getAttr('pose') - thisAgent.getAttr('pose')
            vec_n = np.linalg.norm(vec)
            
            if vec_n!=0:
                vec = vec/vec_n
                
            repel = repel + vec

        if len(thisAgent.getAttr('zor'))!=0:
            repulsion = -1*repel/len(thisAgent.getAttr('zor'))
        else:
            repulsion = -1*repel
        return repulsion
    
    def alignment(self,thisAgent,agents):
        align = np.zeros(2)
        for i in thisAgent.getAttr('zoo'):
            vel_n = np.linalg.norm(agents[i].getAttr('velocity'))
            if vel_n != 0:
                align = align + agents[i].getAttr('velocity')/vel_n

        alignment = align/len(thisAgent.getAttr('zoo'))
        return alignment
    
    def cohesion(self,thisAgent,agents):
        attract = np.zeros(2)
        for i in thisAgent.getAttr('zoa'):
            vec = agents[i].getAttr('pose') - thisAgent.getAttr('pose')
            vec_n = np.linalg.norm(vec)
            if vec_n != 0:
                attract = attract + vec/vec_n
        attraction = attract/len(thisAgent.getAttr('zoa'))
        return attraction

    def requiredYaw(self,thisAgent,agents):
        net_steer = thisAgent.getAttr('velocity')
        if len(thisAgent.zor) != 0:
            net_steer = self.repulsion(thisAgent,agents)
        else:
            if len(thisAgent.getAttr('zoo')) != 0 and len(thisAgent.getAttr('zoa')) == 0:
                net_steer = self.alignment(thisAgent,agents)
            elif len(thisAgent.getAttr('zoo')) == 0 and len(thisAgent.getAttr('zoa')) != 0:
                net_steer = self.cohesion(thisAgent,agents)
            elif len(thisAgent.getAttr('zoo')) != 0 and len(thisAgent.getAttr('zoa')) != 0:
                net_steer = 0.5*(self.alignment(thisAgent,agents)+self.cohesion(thisAgent,agents))
            else:
                net_steer = thisAgent.getAttr('velocity')
                
        vel_norm = np.linalg.norm(net_steer)
        if vel_norm!=0:
            net_steer = net_steer/vel_norm
        return net_steer

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
        
        thisAgent.setAttr('rotationAngle', self.clipper(delta_theta, self.P.getAtt('omega_max')*self.P.getAtt('dt')))
        # thisAgent.setAttr('rotationAngle', self.clipper(delta_theta/self.P.getAtt('dt'), self.P.getAtt('omega_max'))*self.P.getAtt('dt'))

        thisAgent.setAttr('temp_velocity', np.array([[np.cos(thisAgent.getAttr('rotationAngle')), -1*np.sin(thisAgent.getAttr('rotationAngle'))],[np.sin(thisAgent.getAttr('rotationAngle')),np.cos(thisAgent.getAttr('rotationAngle'))]]) @ thisAgent.getAttr('velocity'))
        
        thisAgent.setAttr('temp_yaw', thisAgent.getAttr('yaw') + thisAgent.getAttr('rotationAngle'))
        
        newPose = thisAgent.getAttr('pose') + self.P.getAtt('dt')*thisAgent.getAttr('temp_velocity')
        # if self.P.boundary:
        #     for i in range(2):
        #         if newPose[i]>=self.P.boundary_max:
        #             newPose[i] = newPose[i] - (self.P.boundary_max - self.P.boundary_min)
        #         if newPose[i]<=self.P.boundary_min:
        #             newPose[i] = newPose[i] + (self.P.boundary_max - self.P.boundary_min)
        # if self.P.boundary:
        #     if newPose[0]>=self.P.boundary_max:
        #         newPose[0] = newPose[0] - (self.P.boundary_max - self.P.boundary_min)
        #     if  newPose[1]>=self.P.boundary_max:
        #         newPose[1] = newPose[1] - (self.P.boundary_max - self.P.boundary_min)
        #     if newPose[0]<=self.P.boundary_min:
        #         newPose[0] = newPose[0] + (self.P.boundary_max - self.P.boundary_min)
        #     if newPose[1]<=self.P.boundary_min:
        #         newPose[1] = newPose[1] + (self.P.boundary_max - self.P.boundary_min)
        thisAgent.setAttr('temp_pose', newPose)
    
    def newPose(self,thisAgent):
        thisAgent.setAttr('temp_pose', thisAgent.getAttr('pose') + self.P.getAtt('dt')*thisAgent.getAttr('velocity'))

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

    def getNewStrategy(self, thisAgent, agents):
        # pay_off_diff = np.float128((agent_B.getAttr('payoff')-agent_A.getAttr('payoff')))#abs(np.float128((thatAgent.payoff-thisAgent.payoff)))

        # fermi_fn = 1/(1+np.exp(self.P.getAtt('fermiK') - self.P.getAtt('fermiBeta')*pay_off_diff))

        switch = np.random.uniform(0.0,1.000001)#np.random.choice([0,1], p = [self.P.getAtt('switchProb'),1.0 - self.P.getAtt('switchProb')])#[1.0-fermi_fn,fermi_fn]) # type: ignore
        strategy = cp.copy(thisAgent.getAttr('strategy'))
        if switch<=self.P.getAtt('switchProb'):
            strategy = 0
        else:
            strategy = 1
        thisAgent.setAttr('new_strategy', cp.copy(strategy))
        # best_strategy = thisAgent.getAttr('strategy')
        # best_pay = thisAgent.getAttr('payoff')
        # neighbours = thisAgent.getAttr('zor') #+ thisAgent.getAttr('zoo') + thisAgent.getAttr('zoa')
        # for n in neighbours:
        #     if agents[n].getAttr('payoff') > best_pay:
        #         best_strategy = cp.copy(agents[n].getAttr('strategy'))
        #         best_pay = cp.copy(agents[n].getAttr('payoff'))
        
        # pay_off_diff = np.float128((best_pay-thisAgent.getAttr('payoff')))#abs(np.float128((thatAgent.payoff-thisAgent.payoff)))

        # fermi_fn = 1/(1+np.exp(self.P.getAtt('fermiK') - self.P.getAtt('fermiBeta')*pay_off_diff))

        # switch = np.random.choice([0,1], p = [1.0-fermi_fn,fermi_fn]) # type: ignore
        # strategy = cp.copy(thisAgent.getAttr('strategy'))
        # if switch:
        #     strategy = cp.copy(best_strategy)
        # best_strategy = strategy
        # thisAgent.setAttr('new_strategy', cp.copy(best_strategy))
    
    def update_strategy(self,a):
        a.setAttr('strategy', a.new_strategy)
        
    def game(self,thisAgent, agents):
        # Play the Game here!
        pay = 0.0
        neighbours = thisAgent.getAttr('zor') #+ thisAgent.getAttr('zoo') + thisAgent.getAttr('zoa')
        for n in neighbours:
            pay += self.payoff(thisAgent,agents[n]) # type: ignore
        thisAgent.setAttr('payoff', pay)
        
        
class Simulator:
    def __init__(self,P):
        # Initialize parameters for simulations
        self.P = P
        # Initialize swarm model
        self.interact = CouzinModel(self.P)
        self.game = PrisonerDilema(self.P)
        # Set-up follower agents
        self.interact.createAgents()

    def check_reached(self):
        done = False
        if self.P.getAtt('timer')>self.P.getAtt('end_time'):
            done = True
        return done

    def step(self,surfix):
        # for a in self.interact.agents:
        #     self.interact.neighbours(a,self.interact.agents)
        #     if a.getAttr('strategy'):
        #         self.interact.headingEstimates(a)
        #     else:
        #         self.interact.newPose(a)
        
        for a in self.interact.agents:
            if surfix == '_alld':
                self.interact.neighbours_all(a,self.interact.agents)
            elif surfix == '_zood':
                self.interact.neighbours_zo(a,self.interact.agents)
            elif surfix == '_zord':
                self.interact.neighbours_zr(a,self.interact.agents)
            elif surfix == '_zoad':
                self.interact.neighbours_za(a,self.interact.agents)
            else:
                self.interact.neighbours_all(a,self.interact.agents)
            # if a.getAttr('strategy'):
            self.interact.headingEstimates(a)
            # else:
            #     self.interact.newPose(a)
        
        for a in self.interact.agents:
            self.interact.stateUpdate(a)
        # print(np.round(self.P.getAtt('timer'),decimals = 1)%0.4)
        # if np.round(np.round(self.P.getAtt('timer'),decimals = 1)%0.3, decimals = 1) == 0.0:
        for a in self.interact.agents:#.flatten(): # type: ignore
            self.game.game(a, self.interact.agents)
        
        for a in self.interact.agents:#.flatten(): # type: ignore
            self.game.getNewStrategy(a, self.interact.getAttr('agents'))
        
        for a in self.interact.agents:#.flatten(): # type: ignore
            self.game.update_strategy(a)
