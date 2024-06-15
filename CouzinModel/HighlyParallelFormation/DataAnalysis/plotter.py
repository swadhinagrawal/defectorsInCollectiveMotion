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
        self.ax.set_xlim([-110,110]) # type: ignore
        self.ax.set_ylim([-110,110]) # type: ignore
        
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
        # self.ax.set_xlim([mean[0]-50,mean[0]+50]) # type: ignore
        # self.ax.set_ylim([mean[1]-50,mean[1]+50]) # type: ignore
        # if self.save_scrsht:
        #     self.fig.savefig(self.figpath+str(np.round(self.t,decimals=1))+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
        #     self.save_scrsht = 0
        plt.show()
        plt.pause(0.0000001)
        
        self.fig.canvas.mpl_connect('close_event',onclose)


animate = 1

if animate:
    surfix = '_zoad'
    # surfix = '_zord'
    # surfix = '_zood'
    # surfix = '_alld'
    # surfix = '_test'
    # surfix = '_10n100s1000epzoodlr'
    # surfix = '_50n150szoodlr'
    data_files1 = []#np.array([f for f in get_files_sorted_by_creation(path) if '.pkl' and 'f_50' in f])# and '_33_' not in f])   sorted(os.listdir(path), key=lambda f: os.path.getctime(path+'/'+f))
    n_episodes = 20#100
    # sp = [np.round(0.01*i,decimals=2) for i in range(22,23)]
    sp = [np.round(0.1*i,decimals=1) for i in range(11)]
    for f in sp:
        for i in range(n_episodes):
            if f == 0.1:
                strng = 'P_sp_'+str(f)+'_ep_'+str(i)+'.pkl'
                data_files1.append(strng)


    plt.ion()
    for ep in range(len(data_files1)):
        print(ep)
        plt.close('all')
        df = np.load(path+surfix+'/'+data_files1[ep],allow_pickle=True)
        plot = plotter([df])
        plot.render_init()

        for data_t in df[::10]:
            plot.P = data_t
            plot.world = data_t.world
            plot.bodies_adder()
            plot.renderer()

n_episodes = 20#100#100
max_t = 2001
nAgents = 80#50#50

sp = np.round([0.1*i for i in range(11)],decimals=1) # type: ignore
number_of_colors = len(sp)

color = ['r','g','b','c','m','y','k','orange','purple','brown','pink']
# color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#                 for i in range(number_of_colors)]

# surfix = '_zoad'
# surfix = '_zord'
surfix = '_zood'
# surfix = '_alld'

Log = lg.dataAccess('/results'+surfix)

# No bugs in polarization and misalignment
log_polarization = 0
if log_polarization:
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
    with open(path[:-8]+'/polarization'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)


def BFS(agents):
    '''
    Breadth First Search algorithm for identifying different swarm clusters
    Returns: List of list, each sublist is a different clusters with agents IDs as list elements
    '''
    agents_list = [agents[i].id for i in range(len(agents))]
    components = []
    visited = []
    
    root = 0
    comp_open = [agents_list[0]]
    while len(agents_list)!=0:
        if comp_open[root] not in visited:
            visited.append(comp_open[root])
        for a in agents[comp_open[root]].zor+agents[comp_open[root]].zoo+agents[comp_open[root]].zoa:
            if a not in comp_open and a not in visited:
                comp_open.append(a)
                visited.append(a)

        if root==len(comp_open)-1:
            components.append(comp_open)
            for i in range(len(comp_open)):
                agents_list.pop(agents_list.index(comp_open[i]))
            if len(visited)<len(agents):    
                comp_open = [agents_list[0]]
                root = 0
        elif root < len(comp_open)-1:
            root = root+1
        else:
            break
    # print(components)
    del visited
    del comp_open
    del agents_list
    return components

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


log_group_count = 0
if log_group_count:
    values = []

    for f in tqdm(range(len(sp))):
        group_cnt = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            objects = Log.OneobjExtract(ep = e, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
            for t in range(max_t):
                components = BFS(objects[0][t].world.agents)
                group_cnt[e,t] = len(components)
                del components
            del objects

        values.append(group_cnt)
        
        
    with open(path[:-8]+'/clust'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

log_ang_mom = 0
if log_ang_mom:
    values = []
    std_values = []
    for f in tqdm(range(len(sp))):
        velocity = Log.vectorExtract(name = 'velocity', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        pose = Log.vectorExtract(name = 'pose', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        polarization = np.zeros((n_episodes,nAgents,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                mean_pose = np.zeros(2)
                for a in range(pose.shape[1]):
                    mean_pose += pose[e,a,t]
                mean_pose = mean_pose/float(pose.shape[1])
                
                mean_vector = np.zeros(2)
                for a in range(velocity.shape[1]):
                    vel = velocity[e,a,t]
                    if np.linalg.norm(vel) != 0:
                        vel = vel/np.linalg.norm(vel)
                    r = pose[e,a,t] - mean_pose
                    
                    if np.linalg.norm(r) != 0:
                        r = r/np.linalg.norm(r)
                    polarization[e,a,t] = np.cross(r,vel)
        values.append(polarization)
    with open(path[:-8]+'/ang'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)
 
log_collision = 0
if log_collision:
    values = []
    for f in tqdm(range(len(sp))):
        collide = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            pose,objects = Log.vectorExtract_epi(name = 'pose', ep = e, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
            for t in range(max_t):
                collisions = []
                for a in objects[0][t].world.agents:
                    for n in a.zor:
                        nn_dis = np.linalg.norm(a.pose - objects[0][t].world.agents[n].pose)
                        if nn_dis < 0.2: # type: ignore
                            if objects[0][t].world.agents[n].id not in collisions:
                                collisions.append(objects[0][t].world.agents[n].id)
                            if a.id not in collisions:
                                collisions.append(a.id)
                            
                collide[e,t] = len(collisions)
            del objects
            del pose
                

        values.append(collide)
    with open(path[:-8]+'/colli'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

log_zr_D_frac = 0
if log_zr_D_frac:
    values = []
    for f in tqdm(range(len(sp))):
        pose,objects = Log.vectorExtract1(name = 'pose', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        group_cnt = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                lis = []
                for a in objects[e][t].world.agents:
                    lis += list(a.zor)
                lis = np.unique(lis)
                fract = 0
                for a in lis:
                    if objects[e][t].world.agents[a].strategy == 0:
                        fract += 1

                total = 0
                for a in objects[e][t].world.agents:
                    if a.strategy==0:
                        total += 1
                
                if total!=0:
                    group_cnt[e,t] = fract/total
                else:
                    group_cnt[e,t] = fract

        values.append(group_cnt)
    with open(path[:-8]+'/zor'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

log_za_D_frac = 0
if log_za_D_frac:
    values = []
    for f in tqdm(range(len(sp))):
        pose,objects = Log.vectorExtract1(name = 'pose', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        group_cnt = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                lis = []
                for a in objects[e][t].world.agents:
                    lis += list(a.zoa)
                lis = np.unique(lis)
                fract = 0
                for a in lis:
                    if objects[e][t].world.agents[a].strategy == 0:
                        fract += 1

                total = 0
                for a in objects[e][t].world.agents:
                    if a.strategy==0:
                        total += 1
                
                if total!=0:
                    group_cnt[e,t] = fract/total
                else:
                    group_cnt[e,t] = fract

        values.append(group_cnt)
    with open(path[:-8]+'/zoa'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

log_zo_D_frac = 0
if log_zo_D_frac:
    values = []
    for f in tqdm(range(len(sp))):
        pose,objects = Log.vectorExtract1(name = 'pose', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        group_cnt = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            for t in range(max_t):
                lis = []
                for a in objects[e][t].world.agents:
                    lis += list(a.zoo)
                lis = np.unique(lis)
                fract = 0
                for a in lis:
                    if objects[e][t].world.agents[a].strategy == 0:
                        fract += 1

                total = 0
                for a in objects[e][t].world.agents:
                    if a.strategy==0:
                        total += 1
                
                if total!=0:
                    group_cnt[e,t] = fract/total
                else:
                    group_cnt[e,t] = fract

        values.append(group_cnt)
    with open(path[:-8]+'/zoo'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_energy = 0
if saving_energy:
    values = []
    for f in tqdm(range(len(sp))):
        energy = Log.scalarExtract(name = 'energy', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(energy)
    with open(path[:-8]+'/energy'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_strategy = 0
if saving_strategy:
    values = []
    for f in tqdm(range(len(sp))):
        strategy = Log.scalarExtract(name = 'strategy', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(strategy)
    with open(path[:-8]+'/strategy'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)

saving_payoff = 0
if saving_payoff:
    values = []
    for f in tqdm(range(len(sp))):
        payoff = Log.scalarExtract(name = 'payoff', ep = n_episodes, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
        values.append(payoff)
    with open(path[:-8]+'/extracted_payoff'+surfix+'.npy', 'wb') as saver:
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
    with open(path[:-8]+'/spread'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)

saving_nnD = 0
if saving_nnD:
    values = []
    std_values = []

    for f in tqdm(range(len(sp))):
        
        coop_def_line = np.zeros((n_episodes,max_t))
        std_ = np.zeros((n_episodes,max_t))
        for e in range(n_episodes):
            pose,objects = Log.vectorExtract_epi(name = 'pose', ep = e, nA = nAgents, t = max_t, f = sp[f], initials='/P_sp_')
            for t in range(max_t):
                mean_agents = 0
                for a in range(pose.shape[0]):
                    nearest_nei_dis = 200.0

                    for r in objects[0][t].world.agents:
                        if r.id != a:
                            nn_dis = np.linalg.norm(pose[a,t]- r.pose)
                            if nn_dis < nearest_nei_dis: # type: ignore
                                nearest_nei_dis = nn_dis
                    
                    mean_agents += nearest_nei_dis # type: ignore
                mean_agents = mean_agents/pose.shape[0]
                
                r_spread = np.zeros(pose.shape[0])
                distances = []
                for a in range(pose.shape[0]):
                    # subtractor = np.zeros(2)
                    # if abs(mean_agents[0] - pose[e,a,t][0])> (200)/2.0:
                    #     subtractor[0] = 200
                    # if abs(mean_agents[1] - pose[e,a,t][1])> (200)/2.0:
                    #     subtractor[1] = 200
                    r_spread[a] = np.linalg.norm(mean_agents - pose[a,t])# - subtractor)

                
                coop_def_line[e,t] = mean_agents#np.sum(r_spread).astype(float)/pose.shape[1]
                std_[e,t] = np.std(r_spread) 
            del objects
            del pose
        values.append(coop_def_line)
        std_values.append(std_)
    with open(path[:-8]+'/nnd'+surfix+'.npy', 'wb') as saver:
        for e in values:
            np.save(saver,e)
        for e in std_values:
            np.save(saver,e)

 
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


plt.ioff()
from scipy.signal import savgol_filter

plot_polarization_and_misalignment_ep = 0
if plot_polarization_and_misalignment_ep:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization'+surfix+'.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    fig4,ax4 = plt.subplots()
    fig5,ax5 = plt.subplots()

    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                # for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 0
        ax.set_title('For each iteration')
        ax1.set_title('For each iteration')
        ax2.set_title('Iteration average')
        ax3.set_title('Iteration average')
        ax4.set_title('Polarization > 0.8')
        ax5.set_title('Time to achieve polarization > 0.8')
        
        suc_counter = np.zeros((len(sp),ee[0].shape[0]))
        convergence_time = np.zeros((len(sp),ee[0].shape[0]))
        for e in range(len(ee)):
            for l in range(np.array(ee[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi = ee[e][l]
                std_epi = std_ee[e][l]
                
                for t in range(len(ee[e][l])):
                    if mean_epi[t] > 0.8:
                        suc_counter[e,l] += 1
                        convergence_time[e,l] = np.round(t/10.0, decimals=1)
                        break
                if (i == 4 and l == 0) or (i == 0 and l == 0):
                    ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i], label=Labels[i])
                    ax1.plot(range(ee[e].shape[1]),std_epi,c=color[i], label=Labels[i])
                elif (i == 4 and l != 0) or (i == 0 and l != 0):
                    ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i])
                    ax1.plot(range(ee[e].shape[1]),std_epi,c=color[i])
            
            mean_p = np.mean(ee[e],axis = 0)
            std_p = np.std(ee[e],axis = 0)
            ax2.plot(range(std_ee[e].shape[1]),mean_p,color=color[i],label=Labels[i])
            ax2.fill_between(range(std_ee[e].shape[1]),mean_p-std_p,mean_p+std_p,alpha=0.1,color=color[i])
            mean_stdp = np.mean(np.array(std_ee[e]),axis=0)
            std_stdp = np.std(np.array(std_ee[e]),axis=0)
            
            ax3.plot(range(std_ee[e].shape[1]),mean_stdp,color=color[i],label=Labels[i])
            ax3.fill_between(range(std_ee[e].shape[1]),mean_stdp-std_stdp,mean_stdp+std_stdp,alpha=0.1,color=color[i])
            i += 1
        ax4.bar(np.array(sp)*100,np.sum(suc_counter,axis=1)/n_episodes)
        ax5.bar(np.array(sp)*100,np.mean(convergence_time,axis=1),yerr=np.std(convergence_time))

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Group polarization', fontsize=18)
    ax1.set_xlabel(r'$t$', fontsize=18)
    ax1.set_ylabel('Misalignment', fontsize=18)
    ax2.set_xlabel(r'$t$', fontsize=18)
    ax2.set_ylabel('Group polarization', fontsize=18)
    ax3.set_xlabel(r'$t$', fontsize=18)
    ax3.set_ylabel('Misalignment', fontsize=18)
    ax4.set_xlabel('%D', fontsize=18)
    ax4.set_ylabel('Formation success rate', fontsize=18)
    ax5.set_xlabel('%D', fontsize=18)
    ax5.set_ylabel('Time (in sec)', fontsize=18)
    
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax2.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax3.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/pol'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/misa'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig2.savefig(path_here+'/pol_std'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig3.savefig(path_here+'/misa_std'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig4.savefig(path_here+'/FSR'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig5.savefig(path_here+'/t2r0p8'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_group_count = 0
if plot_group_count:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'clust'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # if i == 4 or i==0:
            #     for epi in range(np.array(ee[e]).shape[0]):
            #         ax1.plot(np.arange(1,np.array(ee[e]).shape[1]),ee[e][epi][1:],color=color[i])
            ax.plot(range(1,ee[e].shape[1]),mean_epi[1:],color=color[i],label=Labels[i])
            mean_epi = np.sum((np.array(ee[e])>1).astype(int),axis=0)/np.array(ee[e]).shape[0]
            mean_t = np.sum(mean_epi[1:])/len(mean_epi[1:])
            ax1.bar(sp[i]*100,mean_t,color=color[i],label=Labels[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Cluster count', fontsize=18)
    ax1.set_xlabel('%D', fontsize=18)
    ax1.set_ylabel('Probability of splitting', fontsize=18)
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),bbox_to_anchor=(1.01, 1),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/clust'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/clust_epi'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_ang_mom = 0
if plot_ang_mom:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'ang'+surfix+'.npy' in f])

    fig,ax = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_agent = np.sum(np.array(ee[e]),axis=1)/np.array(ee[e]).shape[1]
            mean_epi = np.mean(mean_agent,axis=0)
            std_epi = np.std(mean_agent,axis=0)
            ax.plot(range(mean_agent.shape[1]),mean_epi,c=color[i],label=Labels[i])
            ax.fill_between(range(mean_agent.shape[1]), mean_epi- std_epi, mean_epi+ std_epi,color=color[i],alpha=0.1)
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group angular momentum', fontsize=18)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/ang'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()
    
plot_ang_mom_conv = 0
if plot_ang_mom_conv:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'ang'+surfix+'.npy' in f])

    fig,ax = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    ax.plot([-10,200],[0,0],color = 'black')
    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        
        for e in range(len(ee)):
            mean_agent = np.sum(np.array(ee[e]),axis=1)/np.array(ee[e]).shape[1]
            mean_epi = np.mean(mean_agent,axis=0)
            std_epi = np.std(mean_agent,axis=0)
            mean_ = np.mean(mean_epi[-50:])
            std_ = np.mean(std_epi[-50:])
            # ax.bar(sp[e]*100,np.mean(mean_epi[-50:]),yerr=np.mean(std_epi[-50:]),color=color[i],label=Labels[i],width=3,capsize = 3)
            ax.plot([sp[e]*100-0.5,sp[e]*100+0.5],[mean_-std_,mean_-std_],color=color[i])
            ax.plot([sp[e]*100-0.5,sp[e]*100+0.5],[mean_+std_,mean_+std_],color=color[i])
            ax.plot([sp[e]*100,sp[e]*100],[mean_-std_,mean_+std_],color=color[i])
            ax.scatter([sp[e]*100],[mean_],color=color[i])
            # ax.plot(range(mean_agent.shape[1]),mean_epi,c=color[i],label=Labels[i])
            # ax.fill_between(range(mean_agent.shape[1]), mean_epi- std_epi, mean_epi+ std_epi,color=color[i],alpha=0.1)
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Group angular momentum', fontsize=18)
    ax.set_xlim(-5,110)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/ang_conv'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_colli_prob = 0
if plot_colli_prob:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'colli'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # if i == 4 or i==0:
            #     for epi in range(np.array(ee[e]).shape[0]):
            #         ax1.plot(np.arange(1,np.array(ee[e]).shape[1]),ee[e][epi][1:],color=color[i])
            ax.plot(range(1,ee[e].shape[1]),mean_epi[1:],color=color[i],label=Labels[i])
            mean_epi = np.sum((np.array(ee[e])>1).astype(int),axis=0)/np.array(ee[e]).shape[0]
            mean_t = np.sum(mean_epi[1:])/len(mean_epi[1:])
            ax1.bar(sp[i]*100,mean_t,color=color[i],label=Labels[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average collision count', fontsize=18)
    ax1.set_xlabel('%D', fontsize=18)
    ax1.set_ylabel('Probability of collision', fontsize=18)
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),bbox_to_anchor=(1.01, 1),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/colli'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/colli_epi'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_zr_D = 0
if plot_zr_D:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'zor'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # if i == 4 or i==0:
            #     for epi in range(np.array(ee[e]).shape[0]):
            #         ax1.plot(np.arange(1,np.array(ee[e]).shape[1]),ee[e][epi][1:],color=color[i])
            ax.plot(range(ee[e].shape[1]),mean_epi,color=color[i],label=Labels[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('%D at ZOR', fontsize=18)
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    plt.tight_layout()
    fig.savefig(path_here+'/zor'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_za_D = 0
if plot_za_D:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'zoa'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # if i == 4 or i==0:
            #     for epi in range(np.array(ee[e]).shape[0]):
            #         ax1.plot(np.arange(1,np.array(ee[e]).shape[1]),ee[e][epi][1:],color=color[i])
            ax.plot(range(ee[e].shape[1]),mean_epi,color=color[i],label=Labels[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('%D at ZOA', fontsize=18)
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    plt.tight_layout()
    fig.savefig(path_here+'/zoa'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_zo_D = 0
if plot_zo_D:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'zoo'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            # if i == 4 or i==0:
            #     for epi in range(np.array(ee[e]).shape[0]):
            #         ax1.plot(np.arange(1,np.array(ee[e]).shape[1]),ee[e][epi][1:],color=color[i])
            ax.plot(range(ee[e].shape[1]),mean_epi,color=color[i],label=Labels[i])
            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('%D at ZOO', fontsize=18)
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    plt.tight_layout()
    fig.savefig(path_here+'/zoo'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_energy = 0
if plot_energy:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy'+surfix+'.npy' in f and 'bipartite' not in f])
    fig,ax = plt.subplots()
    
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in ee:
            group_e = np.sum(e,axis=1)
            std_grp = savgol_filter(np.std(group_e,axis=0),30,3)
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

plot_energyfew = 0
if plot_energyfew:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy'+surfix+'.npy' in f and 'bipartite' not in f])
    fig,ax = plt.subplots()
    number_of_colors = len(sp)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    for fi in data_files:
        ee = Extract_vals(fi)
        i = 0
        for e in ee[:1]:
            for epi in range(e.shape[0]):
                group_e = np.sum(e[epi],axis=0)
                # std_grp = savgol_filter(np.std(group_e,axis=0),30,3)
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
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_polarization = 0
if plot_polarization:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization'+surfix+'.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    # number_of_colors = len(sp)

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
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization'+surfix+'.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    number_of_colors = len(sp)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 0
        for e in range(0,1):
            for l in range(np.array(ee[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi = ee[e][l]
                ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])

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

plot_ang = 0
if plot_ang:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'ang'+surfix+'.npy' in f])

    fig,ax = plt.subplots()

    # number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    

    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 0
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i],label=Labels[i])

            i += 1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Average group angular momentum', fontsize=18)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/ang'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_R_spread = 0
if plot_R_spread:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'spread'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()

    # number_of_colors = len(sp)

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

plot_nnD = 0
if plot_nnD:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'nnd'+surfix+'.npy' in f])
    
    fig,ax = plt.subplots()

    # number_of_colors = len(sp)

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
    ax.set_ylabel('Average nearest neighbour distance', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/nnd'+surfix+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_convergence_time = 0
if plot_convergence_time:
    number_of_colors = len(sp)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
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
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy'+surfix+'.npy' in f])

    fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    g_sp_new = []

    for fi in data_files:
        ee = Extract_vals(fi)
        g_conv_time = []
        g_conv_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            group_e = savgol_filter(group_e,30,3)
            group_e = np.sum(group_e,axis=0)/group_e.shape[0]
            # group_e = savgol_filter(group_e,30,3)
            # group_e = smooth(group_e,30)
            
            for t in range(group_e.shape[0]-10,window,-1):
                if np.std(group_e[t-window:t+1]) >= np.std(group_e[-window:])+0.7: # type: ignore
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
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization'+surfix+'.npy' in f])

    sp_new = []
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            for t in range(mean_epi.shape[0]-10,window,-1):
                if np.std(mean_epi[(t-window):(t+1)]) >= np.std(mean_epi[-window:])+0.005: # type: ignore
                    conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[int(t-window):t+1]))
                    sp_new.append(sp[e])
                    break

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='blue',label='alignment convergence')
        ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='blue',label='Group polarization convergence')
    ax_vp.set_xlabel('% of defectors', fontsize=18)
    ax_vp.set_ylabel('Polarization', fontsize=18)
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'spread'+surfix+'.npy' in f])
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
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'nnd'+surfix+'.npy' in f])
    fig_vn,ax_vn = plt.subplots()
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
        

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='black',label='nearest neighbour distance convergence')
        ax_vn.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='black',label='nearest neighbour distance convergence')
    ax_vn.set_xlabel('% of defectors', fontsize=18)
    ax_vn.set_ylabel('Distance (in units)', fontsize=18)
    
    fig_vp,ax_vp = plt.subplots()
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'ang'+surfix+'.npy' in f])

    sp_new = []
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            for t in range(mean_epi.shape[0]-10,window,-1):
                if np.std(mean_epi[(t-window):(t+1)]) >= np.std(mean_epi[-window:])+0.005: # type: ignore
                    conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[int(t-window):t+1]))
                    sp_new.append(sp[e])
                    break

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='yellow',label='angular momentum convergence')
        ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='yellow',label='Group angular momentum convergence')
    ax_vp.set_xlabel('% of defectors', fontsize=18)
    ax_vp.set_ylabel('Angular momentum', fontsize=18)
    
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
