#   Decentralized environment with individual state space
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.ion()
figure_on = 1
def onclose(event):
    global figure_on
    figure_on = 0

class Animator:
    def __init__(self):
        print('\nInitializing animator . . .')
        plt.close('all')
        mpl.use('TkAgg')
        
        self.starting = True
        self.fig,self.ax = plt.subplots()
        self.saveFrame = False
        self.figpath = os.path.realpath(os.path.dirname(__file__)) + '/Figs/'
        self.ax.clear()
        self.Pfig,self.Pax = plt.subplots()
        self.Pax.clear()

    def renderer(self, agents, P):
        if self.starting:
            self.starting = False
            self.ax.set_aspect('equal')

            self.ax.set_xlim([P.getAtt('boundary_min'),P.getAtt('boundary_max')]) # type: ignore
            self.ax.set_ylim([P.getAtt('boundary_min'),P.getAtt('boundary_max')]) # type: ignore

            for thisAgent in agents:
                thisAgent.body = self.ax.quiver(thisAgent.pose[0],thisAgent.pose[1],thisAgent.velocity[0],thisAgent.velocity[1],color='blue',linewidths = 0.2)
                self.ax.add_artist(thisAgent.body)

        else:
            for a in agents:
                a.body.set_offsets(np.array([a.pose[0],a.pose[1]]))
                U = P.agent_body_size*np.cos(a.yaw)
                V = P.agent_body_size*np.sin(a.yaw)
                a.body.set_UVC(U,V)
                if a.strategy == 0:
                    a.body.set(color='red')
                if a.strategy == 1:
                    a.body.set(color='blue')
                        
            # for a in np.random.choice(agents.flatten(),size=7,replace=False):
            #     if not isinstance(a.neighbours,type(None)):
            #         # print('changing color')
            #         a.body.set(color='yellow')
            #         for nei in a.neighbours:
            #             agents[tuple(nei)].body.set(color='green')

            self.ax.set_title("Time: "+str(np.round(P.getAtt('timer'),decimals=1))+" steps")
            # if self.save_scrsht:
            #     path = os.path.realpath(os.path.dirname(__file__)) + "/Data/"
            #     self.fig.savefig(self.figpath+str(np.round(self.t,decimals=1))+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
            #     self.save_scrsht = 0
            plt.show()
            # plt.pause(0.0001)
            plt.pause(0.001)
            
            self.fig.canvas.mpl_connect('close_event',onclose)

    def plotter(self, agents, P):
        points = ['o','*']
        color = ['red','blue']
        for a in agents:
            self.Pax.scatter(P.getAtt('timer'),np.arccos(np.clip(np.dot((a.velocity)/np.linalg.norm(a.velocity), (agents[a.neighbours].velocity)/np.linalg.norm(agents[a.neighbours].velocity)),-1,1)),marker = points[a.id], color = color[a.id])
            self.Pax.scatter(P.getAtt('timer'),np.arccos(np.clip(np.dot((a.velocity)/np.linalg.norm(a.velocity), (agents[a.neighbours].velocity)/np.linalg.norm(agents[a.neighbours].velocity)),-1,1)),marker = points[a.id], color = color[a.id])
        self.Pfig.canvas.mpl_connect('close_event',onclose)
        # plt.show()
        # plt.pause(0.0001)
