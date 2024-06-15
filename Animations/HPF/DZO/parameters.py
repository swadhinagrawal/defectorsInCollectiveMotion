import numpy as np
    
# Params
class Params:
    def __init__(self):
        #   Simulation settings
        self.timer = 0.0
        self.Episode = 0
        self.dt = 0.1
        self.end_time = 200.0
        self.render = False          # Bool
        self.logging = bool(1-int(self.render))          # Bool
        self.boundary = False       # Bool  
        self.boundary_min = -100.0
        self.boundary_max = 100.0
        self.agent_body_size = 0.1
        self.nEpisodes = 20#100
        
        #   World settings
        self.Leader_type = 0
        self.Follower_type = 1
        self.Mean_type = 2
        self.Home_type = 3

        # Pay-off matrix
        self.PD_R = 1.0   # Reward
        self.PD_S = 0.0   # Sucker
        self.PD_T = 1.01 #1.01 #(1,2] # Temptation
        self.PD_P = 0.0   # Punishment
        
        # self.payoffMatrix = np.array([[self.PD_R,self.PD_S],[self.PD_T,self.PD_P]])
        
        #   Follower settings
        self.nAgents = 80       #1600
        self.fracDefectors = 0.0        # Fraction of defectors
        self.add_noise_action = False   # Bool
        
        self.fermiK = 0.0       # K
        self.fermiBeta = 0.1   # Selection strength
        # self.omega_max = 5.0*np.pi/180.0
        # self.Rr = 5.0
        # self.Ro = 150.0
        # self.Ra = 155.0
        # self.Rr = 1.0
        # self.Ro = 7.0
        # self.Ra = 20.0
        self.Rr = 1.0
        self.Ro = 18.0
        self.Ra = 20.0
        self.omega_max = 60.0*np.pi/180.0#60.0*np.pi/180.0
        self.fov = 180.0*np.pi/180.0
        self.speed = 5.0
        # self.Rr = 5.0
        # self.Ro = 180.0
        # self.Ra = 195.0
        # self.omega_max = 60.0*np.pi/180.0#60.0*np.pi/180.0
        # self.fov = 170.0*np.pi/180.0
        # self.speed = 30.0
        self.switchProb = 0.1
        self.seed = 0.0
        self.home_r = np.sqrt(self.nAgents)

        
    def getAtt(self, name):
        return getattr(self, name)

    def setAtt(self, name, val):
        return setattr(self, name, val)

if __name__=='__main__':
    p = Params()
    print(p.getAtt('t'))
    p.setAtt('t', 10.0)
    print(p.getAtt('t'))