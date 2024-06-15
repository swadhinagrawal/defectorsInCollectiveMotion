import numpy as np
import matplotlib.pyplot as plt
import os
import logger as lg
from matplotlib import rc
import random
from tqdm import tqdm
import copy as cp


rc('font', weight='bold',size=16)

path = os.path.realpath(os.path.dirname(__file__)) + '/results'
path_here = os.path.realpath(os.path.dirname(__file__))

figure_on = 1
plt.ion()
def onclose(event):
    global figure_on
    figure_on = 0
    
class plotter:
    def __init__(self,data):
        self.P = data[0][0]
        self.figure_on = 1
        self.initialize_plot = 1
        self.world = data[0][0].world
        self.fig,self.ax = plt.subplots()

        self.save_scrsht = 0
        self.plot_timer = None
        self.figpath = os.path.realpath(os.path.dirname(__file__)) + '/frames/'
        self.ax.clear()
        self.bodies = []
    
    def render_init(self):
        self.initialize_plot = 0
        self.bodies = []
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0,5000]) # type: ignore
        self.ax.set_ylim([0,5000]) # type: ignore
        
        # # for s in self.world.home:
        # self.world.home.body = plt.Circle((self.world.home.pose[0],self.world.home.pose[1]),self.world.home.radius,color='green',fill=True,alpha=0.2) # type: ignore
        # self.ax.add_patch(self.world.home.body)

        for thisAgent in self.world.agents:
            thisAgent.body = self.ax.quiver(thisAgent.pose[0],thisAgent.pose[1],thisAgent.velocity[0],thisAgent.velocity[1],color='blue',linewidths = 0.2)
            self.ax.add_artist(thisAgent.body)
        
        # self.bodies.append(self.world.home.body)
        for a in self.world.agents:
            self.bodies.append(a.body)
    
    def bodies_adder(self):
        # self.world.home.body = self.bodies[0]
        for a in range(len(self.world.agents)):
            self.world.agents[a].body = self.bodies[a]
                    
    def renderer(self,Fname):
        def fupdater(a):

            a.body.set_offsets(np.array([a.pose[0],a.pose[1]]))
            U = self.world.P.agent_body_size*np.cos(a.yaw)
            V = self.world.P.agent_body_size*np.sin(a.yaw)
            a.body.set_UVC(U,V)
            if a.strategy == 0:
                a.body.set(color='red')
            if a.strategy == 1:
                a.body.set(color='blue')
                
        # plt.show()
        # plt.pause(1)
        mean = np.array([0.0,0.0])
        for a in self.world.agents:
            # if not isinstance(a.neighbour,type(None)):
            #     # print('changing color')
            #     for nei in a.neighbour:
            #         if nei.id!=525:
            #             nei.body.set(color='green')
            mean += a.pose
            fupdater(a)
        mean /= len(self.world.agents)
        
        # if self.world.P.num_Obs!=0:
        #     for o in self.world.obstacles:
        #         vertices = []
        #         for j in o.vertices_clk:
        #             vertices.append(np.array([j.x,j.y]))
        #         o.body.set_xy(vertices) # type: ignore
        # self.ax.text(110,55,s=)
        self.ax.set_title("Time: "+str(np.round(self.world.P.timer,decimals=1))+" steps")
        # self.ax.set_xlim([mean[0]-80,mean[0]+80]) # type: ignore
        # self.ax.set_ylim([mean[1]-80,mean[1]+80]) # type: ignore
        if np.round(self.world.P.timer,decimals=1) < 2.0:
            self.ax.set_xlim([-10,20]) # type: ignore
            self.ax.set_ylim([-10,20]) # type: ignore
        else:
            self.ax.set_xlim([-50,50]) # type: ignore
            self.ax.set_ylim([-50,50]) # type: ignore
        # if self.save_scrsht:
        #     self.fig.savefig(self.figpath+str(np.round(self.t,decimals=1))+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        #     self.save_scrsht = 0
        plt.show()
        plt.pause(0.000001)
        if np.round(self.world.P.timer,decimals=1)*10%10 == 0.0:
            self.fig.savefig(self.figpath+Fname+'_'+str(np.round(self.world.P.timer,decimals=1))+'.png',format = "png",bbox_inches="tight",pad_inches=0.2,dpi=600)
        
        self.fig.canvas.mpl_connect('close_event',onclose)


animate = 1

if animate:
    data_files1 = []#np.array([f for f in get_files_sorted_by_creation(path) if '.pkl' and 'f_50' in f])# and '_33_' not in f])   sorted(os.listdir(path), key=lambda f: os.path.getctime(path+'/'+f))
    n_episodes = 1
    sp = [0.0,0.3,0.6,0.9]
    for f in sp:
        for i in range(n_episodes):
            strng = 'P_sp_'+str(f)+'_ep_'+str(i)+'.pkl'
            if str(0.9) in strng:
                
                data_files1.append(strng)

    plt.ion()
    for ep in range(len(data_files1)):
        plt.close('all')
        df = np.load(path+'/'+data_files1[ep],allow_pickle=True)
        plot = plotter([df])
        plot.render_init()

        for data_t in df[::]:
            plot.P = data_t
            plot.world = data_t.world
            plot.bodies_adder()
            plot.renderer(data_files1[ep])

