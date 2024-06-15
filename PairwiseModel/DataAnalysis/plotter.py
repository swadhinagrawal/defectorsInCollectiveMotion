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
        self.figpath = os.path.realpath(os.path.dirname(__file__)) + '/Data/'
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
                    
    def renderer(self):
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
        self.ax.set_xlim([mean[0]-200,mean[0]+200]) # type: ignore
        self.ax.set_ylim([mean[1]-200,mean[1]+200]) # type: ignore
        # if self.save_scrsht:
        #     self.fig.savefig(self.figpath+str(np.round(self.t,decimals=1))+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        #     self.save_scrsht = 0
        plt.show()
        plt.pause(0.000001)
        
        self.fig.canvas.mpl_connect('close_event',onclose)


animate = 0

if animate:
    data_files1 = []#np.array([f for f in get_files_sorted_by_creation(path) if '.pkl' and 'f_50' in f])# and '_33_' not in f])   sorted(os.listdir(path), key=lambda f: os.path.getctime(path+'/'+f))
    n_episodes = 100
    sp = [np.round(0.05*i,decimals=1) for i in range(21)]
    for f in sp:
        for i in range(n_episodes):
            if f == 0.3:
                strng = 'P_sp_'+str(f)+'_ep_'+str(i)+'.pkl'
                data_files1.append(strng)
    # plot.render_init()

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
            plot.renderer()


Log = lg.dataAccess('/results')

n_episodes = 100
max_t = 2001
nAgents = 50
# followers = [6,16,40,100]
sp = np.round([0.1*i for i in range(11)],decimals=2) # type: ignore

color = ['r','g','b','c','m','y','k','orange','purple','brown','pink']

saving_energy = 1
if saving_energy:
    values = []
    for f in tqdm(range(len(sp))):
        energy = Log.scalarExtract(name = 'energy', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(energy)
    with open(path[:-8]+'/energy.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_polarization = 1
if saving_polarization:
    values = []
    std_values = []
    for f in tqdm(range(len(sp))):
        velocity = Log.vectorExtract(name = 'velocity', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        polarization = np.zeros((n_episodes,max_t))
        std_polarization = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                mean_vector = np.zeros(2)
                for a in range(velocity.shape[1]):
                    vel = velocity[e,a,t]
                    if np.linalg.norm(vel) != 0:
                        vel = vel/np.linalg.norm(vel)
                    
                    mean_vector += vel
                polarization[e,t] = np.linalg.norm(mean_vector)/float(velocity.shape[1]) # type: ignore
                deviation = 0
                for a in range(velocity.shape[1]):
                    vel = velocity[e,a,t]
                    if np.linalg.norm(vel) != 0:
                        vel = vel/np.linalg.norm(vel)
                    deviation += (np.arccos(np.clip(np.dot(mean_vector/np.linalg.norm(mean_vector),vel),-1,1)))**2
                std_polarization[e,t] = np.sqrt(deviation/float(velocity.shape[1]))/(np.pi)

        values.append(polarization)
        std_values.append(std_polarization)
    with open(path[:-8]+'/polarization.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)

saving_strategy = 0
if saving_strategy:
    values = []
    for f in tqdm(range(len(sp))):
        strategy = Log.scalarExtract(name = 'strategy', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(strategy)
    with open(path[:-8]+'/strategy.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_payoff = 0
if saving_payoff:
    values = []
    for f in tqdm(range(len(sp))):
        payoff = Log.scalarExtract(name = 'payoff', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(payoff)
    with open(path[:-8]+'/extracted_payoff.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_R_spread = 0
if saving_R_spread:
    values = []
    std_values = []

    for f in tqdm(range(len(sp))):
        pose = Log.vectorExtract(name = 'pose', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        coop_def_line = np.zeros((n_episodes,max_t))
        std_ = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                mean_agents = np.zeros(2)
                for a in range(pose.shape[1]):
                    mean_agents += pose[e,a,t]
                    
                mean_agents = mean_agents/pose.shape[1]
                
                r_spread = np.zeros(pose.shape[1])
                distances = []
                for a in range(pose.shape[1]):
                    r_spread[a] = np.linalg.norm(mean_agents-pose[e,a,t])

                
                coop_def_line[e,t] = np.sum(r_spread).astype(float)/pose.shape[1]
                std_[e,t] = np.std(r_spread) 

        values.append(coop_def_line)
        std_values.append(std_)
    with open(path[:-8]+'/spread.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)
 
  
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Extract_vals(file):
    values = []
    with open(path[:-8]+'/'+file, 'rb') as saver:
        for k in range(len(sp)):
            e = np.load(saver,allow_pickle=True)
            values.append(e)
    return values

def Extract_vals_std(file):
    values = []
    std_values = []
    with open(path[:-8]+'/'+file, 'rb') as saver:
        for k in range(len(sp)):
            e = np.load(saver,allow_pickle=True)
            values.append(e)
        for k in range(len(sp)):
            e = np.load(saver,allow_pickle=True)
            std_values.append(e)
    return values, std_values

plt.ioff()
from scipy.signal import savgol_filter
surfix = 'R'
plot_energy = 0
if plot_energy:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy.npy' in f and 'bipartite' not in f])
    fig,ax = plt.subplots()
    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in ee:
            group_e = np.sum(e,axis=1)
            # std_grp = savgol_filter(np.std(group_e,axis=0),30,3)
            group_e = np.sum(group_e,axis=0)/group_e.shape[0]
            group_e = savgol_filter(group_e,30,3)
            # group_e /= -1*np.max(group_e)
            
            ax.plot(range(e.shape[2]),group_e,c=color[i],label=Labels[i])
            # ax.fill_between(range(e.shape[2]),group_e-std_grp,group_e+std_grp,alpha=0.1,color=color[i])
            i += 1
        

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group energy', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/energy'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_polarization = 0
if plot_polarization:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])

            std_epi = np.sum(np.array(std_ee[e]),axis=0)/np.array(std_ee[e]).shape[0]
            ax1.plot(range(std_ee[e].shape[1]),std_epi,color=color[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group polarization', fontsize=18)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.set_xlabel(r'$t$', fontsize=18)
    ax1.set_ylabel('Average alignment deviation', fontsize=18)
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/pol'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_polarizationfewep = 0
if plot_polarizationfewep:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 6
        for e in range(6,7):
            for l in np.random.randint(0,np.array(ee[e]).shape[0],10):
                mean_epi = np.array(ee[e][l])
                ax.plot(range(ee[e].shape[1])[60:220],mean_epi[60:220],c=color[i],label=Labels[i])

            std_epi = np.sum(np.array(std_ee[e]),axis=0)/np.array(std_ee[e]).shape[0]
            ax1.plot(range(std_ee[e].shape[1]),std_epi,color=color[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('group polarization', fontsize=18)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.set_xlabel(r'$t$', fontsize=18)
    ax1.set_ylabel('Average alignment deviation', fontsize=18)
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()


plot_R_spread = 0
if plot_R_spread:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'spread.npy' in f])
    
    fig,ax = plt.subplots()

    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee,std = Extract_vals_std(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # std_epi = np.sum(mean_epi,axis=0)/np.array(e).shape[1]
            ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])
            # print(std_ee[e][:,0])
            # print(np.sum(std_ee[e][:,0])/100)
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group spread', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/spread'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()


plot_convergence_time = 0
if plot_convergence_time:
    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy.npy' in f and 'bipartite' not in f])

    fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    g_sp_new = []

    for fi in data_files:
        ee = Extract_vals(fi)
        g_conv_time = []
        g_conv_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            group_e = np.sum(group_e,axis=0)/group_e.shape[0]
            group_e = savgol_filter(group_e,30,3)
            # group_e = smooth(group_e,30)
            
            for t in range(1,group_e.shape[0]-3):
                if np.std(group_e[t:]) <= 0.1: # type: ignore
                    g_conv_time.append(np.round(t/10.0, decimals=1))
                    g_conv_vals.append(np.mean(group_e[t:]))
                    g_sp_new.append(sp[e])
                    break
                # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
                #     conv_time.append(np.round(1100/10.0, decimals=1))
            # conv_time.append(time_to_conv/len(ee))
        g_conv_time = savgol_filter(g_conv_time,15,2)
        ax.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_time),c='green',label='group energy sigma=0.1 at convergence')
        ax_v.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_vals),c='green',label='Group energy convergence')
    ax_v.set_xlabel('% of defectors', fontsize=18)
    ax_v.set_ylabel('Rotational Energy ('+r'$-\omega^{2}$'+')', fontsize=18)
    
    fig_vp,ax_vp = plt.subplots()
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization.npy' in f])

    sp_new = []
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            for t in range(1,mean_epi.shape[0]-3):
                if np.std(mean_epi[t:]) <= 0.1: # type: ignore
                    conv_time.append(np.round(t/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[t:]))
                    sp_new.append(sp[e])
                    break

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='blue',label='alignment sigma=0.1 at convergence')
        ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='blue',label='Group polarization convergence')
    ax_vp.set_xlabel('% of defectors', fontsize=18)
    ax_vp.set_ylabel('Polarization', fontsize=18)
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'spread.npy' in f])
    fig_vs,ax_vs = plt.subplots()
    sp_new = []
    for fi in data_files:
        ee,std = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            mean_epi = mean_epi/np.max(mean_epi)
            # print(np.max(np.array(ee[e])))
            # print(mean_epi)
            for t in range(mean_epi.shape[0]-3):
                if np.std(mean_epi[t:]) <= 0.1: # type: ignore
                    conv_time.append(np.round(t/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[t:]))
                    sp_new.append(sp[e])
                    break
                # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
                #     conv_time.append(np.round(1100/10.0, decimals=1))
            # conv_time.append(time_to_conv/len(ee))
        

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='red',label='cohesion sigma=0.1 at convergence')
        ax_vs.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='red',label='Group cohesion convergence')
    ax_vs.set_xlabel('% of defectors', fontsize=18)
    ax_vs.set_ylabel('Distance (in units)', fontsize=18)
    
    ax.set_xlabel('% of defectors', fontsize=18)
    ax.set_ylabel('Time (in sec)', fontsize=18)
    # ax.set_title('For 98% cohesion convergence', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_convergence_time1 = 0
if plot_convergence_time1:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy.npy' in f and 'bipartite' not in f])

    fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    g_sp_new = []

    for fi in data_files:
        ee = Extract_vals(fi)
        g_conv_time = []
        g_conv_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            group_e = np.sum(group_e,axis=0)/group_e.shape[0]
            # group_e = savgol_filter(group_e,30,3)
            # group_e = smooth(group_e,30)
            
            for t in range(group_e.shape[0]-10,window,-1):
                if np.std(group_e[t-window:t+1]) >= np.std(group_e[-window:])+0.01: # type: ignore
                    g_conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    g_conv_vals.append(np.mean(group_e[t-window:t+1]))
                    g_sp_new.append(sp[e])
                    break
                # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
                #     conv_time.append(np.round(1100/10.0, decimals=1))
            # conv_time.append(time_to_conv/len(ee))
        # g_conv_time = savgol_filter(g_conv_time,15,2)
        ax.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_time),c='green',label='group energy convergence')
        ax_v.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_vals),c='green',label='Group energy convergence')
    ax_v.set_xlabel('% of defectors', fontsize=18)
    ax_v.set_ylabel('Rotational Energy ('+r'$-\omega^{2}$'+')', fontsize=18)
    
    fig_vp,ax_vp = plt.subplots()
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization.npy' in f])

    sp_new = []
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            for t in range(mean_epi.shape[0]-10,window,-1):
                if np.std(mean_epi[(t-window):(t+1)]) >= np.std(mean_epi[-window:])+0.01: # type: ignore
                    conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[int(t-window):t+1]))
                    sp_new.append(sp[e])
                    break

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='blue',label='alignment convergence')
        ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='blue',label='Group polarization convergence')
    ax_vp.set_xlabel('% of defectors', fontsize=18)
    ax_vp.set_ylabel('Polarization', fontsize=18)
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'spread.npy' in f])
    fig_vs,ax_vs = plt.subplots()
    sp_new = []
    for fi in data_files:
        ee,std = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # mean_epi = mean_epi/np.max(mean_epi)
            # print(np.max(np.array(ee[e])))
            # print(mean_epi)
            for t in range(mean_epi.shape[0]-10,window,-1):
                if np.std(mean_epi[t-window:t+1]) >= np.std(mean_epi[-window:])+0.01: # type: ignore
                    conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[t-window:t+1]))
                    sp_new.append(sp[e])
                    break
                # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
                #     conv_time.append(np.round(1100/10.0, decimals=1))
            # conv_time.append(time_to_conv/len(ee))
        

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='red',label='cohesion convergence')
        ax_vs.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='red',label='Group cohesion convergence')
    ax_vs.set_xlabel('% of defectors', fontsize=18)
    ax_vs.set_ylabel('Distance (in units)', fontsize=18)
    
    ax.set_xlabel('% of defectors', fontsize=18)
    ax.set_ylabel('Time (in sec)', fontsize=18)
    # ax.set_title('For 98% cohesion convergence', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/t2c'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()


plot_strategy = 0
if plot_strategy:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'strategy.npy' in f])
    followers = [100]
    sp = [0.1*i for i in range(11)]
    def Extract_file(file):
        values = []
        with open(path[:-8]+'/'+file, 'rb') as saver:
            for k in range(len(sp)):
                e = np.load(saver,allow_pickle=True)
                values.append(e)
        return values
    
    plt.ioff()
    fig,ax = plt.subplots()

    
    number_of_colors = len(sp)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['%D = '+str(np.round(i,decimals=2)*100) for i in sp]
    

    for fi in data_files:
        ee = Extract_file(fi)
        i = 0
        for e in ee:
            collaborator_fraction_epi = np.sum(e,axis=1)/e.shape[1]
            avg_collaborator_fraction = np.sum(collaborator_fraction_epi,axis=0)/e.shape[0]
            std_collaboration = np.std(collaborator_fraction_epi,axis=0)
            ax.plot(range(e.shape[2]),avg_collaborator_fraction,c=color[i],label=Labels[i])
            ax.fill_between(range(e.shape[2]),avg_collaborator_fraction-std_collaboration,avg_collaborator_fraction+std_collaboration,alpha=0.2,color=color[i])
            # ax[1].plot(range(e.shape[2]),indi_avg_e,c=color[i],label=Labels[i])

            i += 1
        

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average fraction of collaborators', fontsize=18)
    # ax[1].set_xlabel(r'$t$', fontsize=18)
    # ax[1].set_ylabel('Average individual energy', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    # ax[1].legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_payoff = 0
if plot_payoff:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if '.npy' in f and 'payoff.' in f])
    followers = [6,16,40,100]
    def Extract_file(file):
        values = []
        with open(path[:-8]+'/'+file, 'rb') as saver:
            for k in range(len(followers)):
                e = np.load(saver,allow_pickle=True)
                values.append(e)
        return values
    
    plt.ioff()
    fig,ax = plt.subplots(1,2)

    
    number_of_colors = len(followers)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['F'+str(i) for i in followers]
    

    for fi in data_files:
        ee = Extract_file(fi)
        i = 0
        for e in ee:
            mean_epi = np.sum(e,axis=0)/e.shape[0]
            group_e = np.sum(mean_epi,axis=0)
            std_grp = np.std(np.sum(e,axis=1),axis=0)
            indi_avg_e = np.sum(mean_epi,axis=0)/e.shape[1] # num_agents
            std_indi = np.std(mean_epi,axis=0)
            
            ax[0].plot(range(e.shape[2]),group_e,c=color[i],label=Labels[i])
            ax[0].fill_between(range(e.shape[2]),group_e-std_grp,group_e+std_grp,alpha=0.2,color=color[i])
            ax[1].plot(range(e.shape[2]),indi_avg_e,c=color[i],label=Labels[i])
            ax[1].fill_between(range(e.shape[2]),indi_avg_e-std_indi,indi_avg_e+std_indi,alpha=0.2,color=color[i])

            i += 1
        

    ax[0].set_xlabel(r'$t$', fontsize=18)
    ax[0].set_ylabel('Average group payoff', fontsize=18)
    ax[1].set_xlabel(r'$t$', fontsize=18)
    ax[1].set_ylabel('Average individual payoff', fontsize=18)

    ax[0].legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    ax[1].legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

def Extract_ang_mom(files):
    values = []
    std_values = []
    followers = [6,16,40,100]
    n_episodes = 100
    max_t = 2001
    for f in tqdm(range(len(followers))):
        polarization = np.zeros((n_episodes,max_t))
        std_polarization = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            par = np.load(path+'/'+files[f*n_episodes+e],allow_pickle=True)
            tt = 0
            for t in par:
                mean_vector = np.zeros(2)
                for a in t.world.F_agents:
                    vel = a.velocity
                    if np.linalg.norm(vel) != 0:
                        vel = vel/np.linalg.norm(vel)
                    
                    mean_vector += np.cross(a.r_com,vel)
                polarization[e,tt] = np.linalg.norm(mean_vector)/t.world.P.num_F
                deviation = 0
                for a in t.world.F_agents:
                    vel = a.velocity
                    if np.linalg.norm(vel) != 0:
                        vel = vel/np.linalg.norm(vel)
                    # print(np.linalg.norm(vel))
                    ang_mom = np.cross(a.r_com,vel)
                    deviation += (polarization[e,tt] - np.linalg.norm(ang_mom))**2
                std_polarization[e,tt] = np.sqrt(deviation/t.world.P.num_F)
                tt += 1
        values.append(polarization)
        std_values.append(std_polarization)
    with open(path[:-8]+'/extracted_ang_mom.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)

saving_ang_mom = 0
if saving_ang_mom:
    data_files1 = []#np.array([f for f in get_files_sorted_by_creation(path) if '.pkl' and 'f_50' in f])# and '_33_' not in f])   sorted(os.listdir(path), key=lambda f: os.path.getctime(path+'/'+f))
    n_episodes = 100
    for f in [6,16,40,100]:
        for i in range(n_episodes):
            strng = 'P_f_'+str(int(f))+'_ep_'+str(i)+'.pkl'
            data_files1.append(strng)
    Extract_ang_mom(data_files1)

plot_ang_mom = 0
if plot_ang_mom:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if '.npy' in f and 'ang_mom.' in f])
    followers = [6,16,40,100]
    def Extract_file(file):
        values = []
        std_values = []
        with open(path[:-8]+'/'+file, 'rb') as saver:
            for k in range(len(followers)):
                e = np.load(saver,allow_pickle=True)
                values.append(e)
            for k in range(len(followers)):
                e = np.load(saver,allow_pickle=True)
                std_values.append(e)
        return values, std_values
    
    plt.ioff()
    fig,ax = plt.subplots()

    
    number_of_colors = len(followers)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['F'+str(i) for i in followers]
    

    for fi in data_files:
        ee,std_ee = Extract_file(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # std_epi = np.sum(mean_epi,axis=0)/np.array(e).shape[1]
            ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])
            # print(std_ee[e][:,0])
            # print(np.sum(std_ee[e][:,0])/100)
            std_epi = np.sum(np.array(std_ee[e]),axis=0)/np.array(std_ee[e]).shape[0]
            ax.fill_between(range(std_ee[e].shape[1]),np.clip(mean_epi-std_epi,0,1),mean_epi+std_epi,alpha=0.1,color=color[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group angular momentum', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

def Extract_coop_def_line(files):
    values = []
    followers = [6,16,40,100]
    n_episodes = 100
    max_t = 2001
    for f in tqdm(range(len(followers))):
        coop_def_line = np.zeros((n_episodes,max_t,2))
        for e in range(n_episodes):
            par = np.load(path+'/'+files[f*n_episodes+e],allow_pickle=True)
            tt = 0
            for t in par:
                mean_coop = np.zeros(2)
                mean_defe = np.zeros(2)
                counter_coop = 0
                counter_defe = 0
                for a in t.world.F_agents:
                    if a.strategy == 0:
                        mean_defe += np.array(a.pose)
                        counter_defe += 1
                    if a.strategy == 1:
                        mean_coop += np.array(a.pose)
                        counter_coop += 1
                
                if counter_defe!=0:
                    mean_defe = mean_defe/counter_defe
                if counter_coop!=0:
                    mean_coop = mean_coop/counter_coop
                vector = mean_coop-mean_defe
                if counter_defe == 0 or counter_coop == 0:
                    vector = np.zeros(2)
                coop_def_line[e,tt] = vector
                tt += 1
        values.append(coop_def_line)
    with open(path[:-8]+'/extracted_polarization.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_coop_def_line = 0
if saving_coop_def_line:
    data_files1 = []#np.array([f for f in get_files_sorted_by_creation(path) if '.pkl' and 'f_50' in f])# and '_33_' not in f])   sorted(os.listdir(path), key=lambda f: os.path.getctime(path+'/'+f))
    n_episodes = 100
    for f in [6,16,40,100]:
        for i in range(n_episodes):
            strng = 'P_f_'+str(int(f))+'_ep_'+str(i)+'.pkl'
            data_files1.append(strng)
    Extract_coop_def_line(data_files1)

plot_coop_def_line = 0
if plot_coop_def_line:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if '.npy' in f and 'polarization.' in f])
    followers = [6,16,40,100]
    def Extract_files(file):
        values = []
        std_values = []
        with open(path[:-8]+'/'+file, 'rb') as saver:
            for k in range(len(followers)):
                e = np.load(saver,allow_pickle=True)
                values.append(e)
        return values
    
    plt.ioff()
    fig,ax = plt.subplots()

    number_of_colors = len(followers)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['F'+str(i) for i in followers]
    

    for fi in data_files:
        ee = Extract_files(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # std_epi = np.sum(mean_epi,axis=0)/np.array(e).shape[1]
            ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])
            # print(std_ee[e][:,0])
            # print(np.sum(std_ee[e][:,0])/100)
            std_epi = np.sum(np.array(std_ee[e]),axis=0)/np.array(std_ee[e]).shape[0]
            ax.fill_between(range(std_ee[e].shape[1]),np.clip(mean_epi-std_epi,0,1),mean_epi+std_epi,alpha=0.1,color=color[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group polarization', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()
