import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import matplotlib as mpl
import json

rc('font',size=24)

path = os.path.realpath(os.path.dirname(__file__)) + '/results'
path_here = os.path.realpath(os.path.dirname(__file__))


n_episodes = 20#100
max_t = 2001
N = [20,50,80]
# followers = [6,16,40,100]
sp = np.round([0.1*i for i in range(11)],decimals=2) # type: ignore

color = ['#E48888','#31E8C9','#66B2FF','#FF99CC']#['r','g','b','c','m','y','k','orange','purple','brown','pink']
colorD = ['r','g','b','c','m','y','k','orange','purple','brown','pink']
def Extract_vals(file):
    values = []
    with open(path[:-8]+'/'+file, 'rb') as saver:
        for k in range(len(sp)):
            e = np.load(saver,allow_pickle=True)
            values.append(e)
    return values

from scipy.signal import savgol_filter

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

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

plot_convergence_time1 = 0
if plot_convergence_time1:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$']
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f])

    fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    sigma = [0.02, 0.4, 0.01, 0.7]
    ii = 0
    for fi in data_files:
        g_sp_new = []
        ee = Extract_vals(fi)
        g_conv_time = []
        g_conv_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            group_e = np.sum(group_e,axis=0)/group_e.shape[0]
            # group_e = savgol_filter(group_e,30,3)
            # group_e = smooth(group_e,30)
            
            for t in range(group_e.shape[0],window,-1):
                if np.std(group_e[t-window:t+1]) >= np.std(group_e[-window:])+sigma[ii]: # type: ignore
                    g_conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    g_conv_vals.append(np.mean(group_e[t-window:t+1]))
                    g_sp_new.append(sp[e])
                    break
                # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
                #     conv_time.append(np.round(1100/10.0, decimals=1))
            # conv_time.append(time_to_conv/len(ee))
        # g_conv_time = savgol_filter(g_conv_time,15,2)
        ax.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_time),c=color[ii],label=Labels[ii])
        ax_v.plot((np.round(np.array(g_sp_new),decimals=2)*100).astype(int),np.array(g_conv_vals),c=color[ii],label=Labels[ii])
        ii += 1
    ax_v.set_xlabel('% of defectors', fontsize=24)
    ax_v.set_ylabel('Rotational Energy ('+r'$-\omega^{2}$'+')', fontsize=24)
    
    # fig_vp,ax_vp = plt.subplots()
    # data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization.npy' in f])

    # sp_new = []
    # for fi in data_files:
    #     ee,std_ee = Extract_vals_std(fi)
    #     conv_time = []
    #     conv_val = []
    #     for e in range(len(ee)):
    #         mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
    #         for t in range(mean_epi.shape[0]-10,window,-1):
    #             if np.std(mean_epi[(t-window):(t+1)]) >= np.std(mean_epi[-window:])+0.01: # type: ignore
    #                 conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
    #                 conv_val.append(np.mean(mean_epi[int(t-window):t+1]))
    #                 sp_new.append(sp[e])
    #                 break

    #     ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='blue',label='alignment sigma=0.01 at convergence')
    #     ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='blue',label='Group polarization convergence')
    # ax_vp.set_xlabel('% of defectors', fontsize=24)
    # ax_vp.set_ylabel('Polarization', fontsize=24)
    
    # data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'spread.npy' in f])
    # fig_vs,ax_vs = plt.subplots()
    # sp_new = []
    # for fi in data_files:
    #     ee,std = Extract_vals_std(fi)
    #     conv_time = []
    #     conv_val = []
    #     for e in range(len(ee)):
    #         mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
    #         # mean_epi = mean_epi/np.max(mean_epi)
    #         # print(np.max(np.array(ee[e])))
    #         # print(mean_epi)
    #         for t in range(mean_epi.shape[0]-10,window,-1):
    #             if np.std(mean_epi[t-window:t+1]) >= np.std(mean_epi[-window:])+0.01: # type: ignore
    #                 conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
    #                 conv_val.append(np.mean(mean_epi[t-window:t+1]))
    #                 sp_new.append(sp[e])
    #                 break
    #             # elif mean_epi[t] < 0.98 and t == mean_epi.shape[0]-1:
    #             #     conv_time.append(np.round(1100/10.0, decimals=1))
    #         # conv_time.append(time_to_conv/len(ee))
        

    #     ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c='red',label='cohesion sigma=0.01 at convergence')
    #     ax_vs.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c='red',label='Group cohesion convergence')
    # ax_vs.set_xlabel('% of defectors', fontsize=24)
    # ax_vs.set_ylabel('Distance (in units)', fontsize=24)
    
    ax.set_xlabel('% of defectors', fontsize=24)
    ax.set_ylabel('Time (in sec)', fontsize=24)
    # # ax.set_title('For 98% cohesion convergence', fontsize=24)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/energy_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_convergence_time_after_pol = 0
if plot_convergence_time_after_pol:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$']
    
    

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(10,8))
    quad = {'0':(0,0),'1':(0,1),'2':(1,0),'3':(1,1)}
    ax2 = ax_v[quad[str(0)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    ax3 = ax_v[quad[str(1)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    ax4 = ax_v[quad[str(2)]].inset_axes([0.5, 0.2, 0.5, 0.4])
    ax5 = ax_v[quad[str(3)]].inset_axes([0.4, 0.4, 0.5, 0.4])
    axes = [ax2,ax3,ax4,ax5]
    nC = 0
    for nAgents in N:
        data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f and str(nAgents)+'A' in f])
        data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])
        ii = 0
        for fi in range(len(data_files)):
            ee = Extract_vals(data_files[fi])
            ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])
            suc_counter = np.zeros((len(sp),ee[0].shape[0]))
            
            g_conv_vals = []
            g_conv_std_vals = []
            for e in range(len(ee)):
                group_e = np.sum(ee[e],axis=1)
                
                pol_grp = np.zeros(np.array(group_e).shape)
                unpol_grp = np.zeros(np.array(group_e).shape)
                energies = np.zeros(group_e.shape[0])
                
                
                for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                    mean_epi_ = ee_pol[e][l]
                    tt = 0
                    for t in range(len(ee_pol[e][l])):
                        
                        if mean_epi_[t] > 0.8:
                            flag = 1
                            suc_counter[e,l] += 1
                            energies[l] = np.mean(group_e[l,t:])
                            break
                        else:
                            flag = 0
                            tt = t
                        energies[l] = np.mean(group_e[l,tt:])
                    if flag:
                        pol_grp[l,:] = 1
                    else:
                        unpol_grp[l,:] = 1
                        # energies[l] = np.mean(group_e[l,tt-10:tt])
                    # energies[l] = group_e[l,tt]
                    
                
                # polarised = energies*pol_grp[:,-1]
                # unpolarised = energies*unpol_grp[:,-1]
                
                
                g_conv_vals.append(np.mean(energies))
                g_conv_std_vals.append(np.std(energies))

            # ax_v.errorbar((np.round(np.array(sp),decimals=1)*100+fi*1.5),(np.array(g_conv_vals)),yerr=g_conv_std_vals,color=color[fi],label=Labels[fi])
            std_ = np.std(suc_counter,axis=1)
            mean_ = np.mean(suc_counter,axis=1)
            for i in range(len(sp)):
                if mean_[i]>0.8:
                    g_conv_std_vals_max = g_conv_std_vals[i]
                    if g_conv_std_vals[i] + g_conv_vals[i]>0:
                        g_conv_std_vals_max = abs(g_conv_vals[i])
                    if i == 0:
                        if nC==0:
                            ax_v[quad[str(fi)]].bar(-10,0,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                        if fi == 3 and nC==0:
                            ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='N = '+str(N[0]),width=1)
                            ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='N = '+str(N[1]),width=1)    
                    if nC==0:
                        ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    else:
                        ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = color[fi],edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    
                    if nC==0:
                        axes[fi].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    # else:    
                    #     axes[fi].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[abs(np.clip(g_conv_vals[i]-g_conv_std_vals[i],-10000,0))],[abs(np.clip(g_conv_vals[i]+g_conv_std_vals[i],-10000,0))]],facecolor = color[fi],edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                # if mean_[i]>0.8:
                #     ax_v.scatter(np.array(sp)[i]*100+fi*1.5,g_conv_vals[i],color=color[fi])
                #     ax_v.plot([sp[i]*100.0+fi*1.5-0.5,sp[i]*100.0+fi*1.5+0.5],[np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0),np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0)],color=color[fi],linestyle='-')
                #     ax_v.plot([sp[i]*100.0+fi*1.5-0.5,sp[i]*100.0+fi*1.5+0.5],[np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0),np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0)],color=color[fi],linestyle='-')
                #     ax_v.plot([sp[i]*100.0+fi*1.5,sp[i]*100.0+fi*1.5],[np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0),np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0)],color=color[fi])
                
            ii += 1
        nC += 1
    fig_v.supxlabel('Fraction of defectors', fontsize=24)
    fig_v.supylabel('Mean Energy ('+r'$\frac{\Sigma_{t= p>0.8}^{200} e(t)}{\Sigma_{t= p>0.8}^{200}\delta t}$'+')', fontsize=24)
    fig_v.suptitle('Above Cut-off = 0.8', fontsize=24)
    ax_v[0,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[0,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[1,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[1,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[quad[str(0)]].indicate_inset_zoom(ax2, edgecolor="black")
    ax_v[quad[str(1)]].indicate_inset_zoom(ax3, edgecolor="black")
    ax_v[quad[str(2)]].indicate_inset_zoom(ax4, edgecolor="black")
    ax_v[quad[str(3)]].indicate_inset_zoom(ax5, edgecolor="black")
    ax_v[0,0].set_ylim(-60,0)
    ax_v[0,1].set_ylim(-60,0)
    ax_v[1,0].set_ylim(-60,0)
    ax_v[1,1].set_ylim(-60,0)
    ax2.set_ylim(-2,0) 
    ax3.set_ylim(-2,0) 
    ax4.set_ylim(-2,0) 
    ax5.set_ylim(-2,0) 
    # ax.set_xlabel('% of defectors', fontsize=24)
    # ax.set_ylabel('Time (in sec)', fontsize=24)
    # # ax.set_title('For 98% cohesion convergence', fontsize=24)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/energy_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_convergence_time_bef_pol = 0
if plot_convergence_time_bef_pol:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$']
    
    

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(10,8))
    quad = {'0':(0,0),'1':(0,1),'2':(1,0),'3':(1,1)}
    
    ax2 = ax_v[quad[str(0)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    ax3 = ax_v[quad[str(1)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    ax4 = ax_v[quad[str(2)]].inset_axes([0.5, 0.2, 0.5, 0.4])
    ax5 = ax_v[quad[str(3)]].inset_axes([0.4, 0.4, 0.5, 0.4])
    axes = [ax2,ax3,ax4,ax5]
    nC = 0
    for nAgents in N:
        data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f and str(nAgents)+'A' in f])
        data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])
        ii = 0
        for fi in range(len(data_files)):
            ee = Extract_vals(data_files[fi])
            ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])
            suc_counter = np.zeros((len(sp),ee[0].shape[0]))
            
            g_conv_vals = []
            g_conv_std_vals = []
            for e in range(len(ee)):
                group_e = np.sum(ee[e],axis=1)
                
                pol_grp = np.zeros(np.array(group_e).shape)
                unpol_grp = np.zeros(np.array(group_e).shape)
                energies = np.zeros(group_e.shape[0])
                
                
                for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                    mean_epi_ = ee_pol[e][l]
                    tt = 0
                    for t in range(len(ee_pol[e][l])):
                        
                        if mean_epi_[t] > 0.8:
                            flag = 1
                            suc_counter[e,l] += 1
                            energies[l] = np.sum(group_e[l,:])*0.1
                            break
                        else:
                            flag = 0
                            tt = t
                        energies[l] = np.sum(group_e[l,:])*0.1
                    if flag:
                        pol_grp[l,:] = 1
                    else:
                        unpol_grp[l,:] = 1
                        # energies[l] = np.mean(group_e[l,tt-10:tt])
                    # energies[l] = group_e[l,tt]
                    
                
                # polarised = energies*pol_grp[:,-1]
                # unpolarised = energies*unpol_grp[:,-1]
                
                
                g_conv_vals.append(np.mean(energies))
                g_conv_std_vals.append(np.std(energies))

            # ax_v.errorbar((np.round(np.array(sp),decimals=1)*100+fi*1.5),(np.array(g_conv_vals)),yerr=g_conv_std_vals,color=color[fi],label=Labels[fi])
            std_ = np.std(suc_counter,axis=1)
            mean_ = np.mean(suc_counter,axis=1)
            for i in range(len(sp)):
                if mean_[i]>0.8:
                    g_conv_std_vals_max = g_conv_std_vals[i]
                    if g_conv_std_vals[i] + g_conv_vals[i]>0:
                        g_conv_std_vals_max = abs(g_conv_vals[i])
                    if i == 0:
                        if nC==0:
                            ax_v[quad[str(fi)]].bar(-10,0,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                        if fi == 3 and nC==0:
                            ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='N = '+str(N[0]),width=1)
                            ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='N = '+str(N[1]),width=1)    
                    if nC==0:
                        ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    else:
                        ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = color[fi],edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                        
                    if nC==0:
                        axes[fi].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    # else:    
                    #     axes[fi].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[abs(np.clip(g_conv_vals[i]-g_conv_std_vals[i],-10000,0))],[abs(np.clip(g_conv_vals[i]+g_conv_std_vals[i],-10000,0))]],facecolor = color[fi],edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                # if mean_[i]>0.8:
                #     ax_v.scatter(np.array(sp)[i]*100+fi*1.5,g_conv_vals[i],color=color[fi])
                #     ax_v.plot([sp[i]*100.0+fi*1.5-0.5,sp[i]*100.0+fi*1.5+0.5],[np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0),np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0)],color=color[fi],linestyle='-')
                #     ax_v.plot([sp[i]*100.0+fi*1.5-0.5,sp[i]*100.0+fi*1.5+0.5],[np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0),np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0)],color=color[fi],linestyle='-')
                #     ax_v.plot([sp[i]*100.0+fi*1.5,sp[i]*100.0+fi*1.5],[np.clip(g_conv_vals[i]-g_conv_std_vals[i],-1000,0),np.clip(g_conv_std_vals[i]+g_conv_vals[i],-1000,0)],color=color[fi])
                
            ii += 1
        nC += 1
    fig_v.supxlabel('Fraction of defectors', fontsize=24)
    fig_v.supylabel('Total Energy ('+r'$\Sigma_{t=0}^{200} e(t)\delta t$'+')', fontsize=24)
    fig_v.suptitle('Above Cut-off = 0.8', fontsize=24)
    ax_v[0,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[0,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[1,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[1,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    ax_v[quad[str(0)]].indicate_inset_zoom(ax2, edgecolor="black")
    ax_v[quad[str(1)]].indicate_inset_zoom(ax3, edgecolor="black")
    ax_v[quad[str(2)]].indicate_inset_zoom(ax4, edgecolor="black")
    ax_v[quad[str(3)]].indicate_inset_zoom(ax5, edgecolor="black")
    
    
    ax2.set_ylim(-200,0) 
    ax3.set_ylim(-200,0) 
    ax4.set_ylim(-200,0) 
    ax5.set_ylim(-200,0) 
    # ax.set_xlabel('% of defectors', fontsize=24)
    # ax.set_ylabel('Time (in sec)', fontsize=24)
    # # ax.set_title('For 98% cohesion convergence', fontsize=24)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/bef_energy_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()


def set_axis_style(ax, labels,ylab,labe,col):
    ax.set_xticks(np.array(labels).astype(int),labels=np.array(labels).astype(int))
    ax.set_xlim(min(labels)-10, max(labels) + 10)
    
    # ax.set_xlabel('Proportion of defectors')
    # ax.set_facecolor('#D4D4D4')
def set_axis_leg(ax, labe,col):    
    ax.plot([],[],color='#8E006B',linestyle='--',label=r'$\bf{\mu}$')
    # ax.plot([],[],color='#8E006B',linestyle=':',label='Median')
    rec = mpl.patches.Rectangle((0,0),0,0,fc=col, edgecolor = 'none', linewidth = 0,label=labe,alpha=0.2)
    ax.add_artist(rec)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1,bbox_to_anchor=(1.2, 1))
    
    
plot_energy_after_pol = 1
if plot_energy_after_pol:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'${\bf Z_{all}}$','D-'+r'${\bf Z_{a}}$','D-'+r'${\bf Z_{o}}$','D-'+r'${\bf Z_{r}}$']
    
    

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(10,10))
    quad = {'0':0,'1':1,'2':2,'3':3}
    # ax2 = ax_v[quad[str(0)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    # ax3 = ax_v[quad[str(1)]].inset_axes([0.4, 0.2, 0.5, 0.4])
    # ax4 = ax_v[quad[str(2)]].inset_axes([0.5, 0.2, 0.5, 0.4])
    # ax5 = ax_v[quad[str(3)]].inset_axes([0.4, 0.4, 0.5, 0.4])
    # axes = [ax2,ax3,ax4,ax5]
    nC = 0
    for nAgents in N:
        data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f and str(nAgents)+'A' in f])
        data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])
        ii = 0
        for fi in range(len(data_files)):
            ee = Extract_vals(data_files[fi])
            ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])
            suc_counter = np.zeros((len(sp),ee[0].shape[0]))
            pol_at_conv = np.zeros((len(sp),ee[0].shape[0]))
            g_conv_vals = []
            g_conv_std_vals = []
            g_conv_vals_all = []
            for e in range(len(ee)):
                group_e = np.sum(ee[e],axis=1)
                
                pol_grp = np.zeros(np.array(group_e).shape)
                unpol_grp = np.zeros(np.array(group_e).shape)
                energies = np.zeros(group_e.shape[0])
                
                
                for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                    mean_epi_ = ee_pol[e][l]
                    tt = 0
                    for t in range(len(ee_pol[e][l])):
                        
                        if mean_epi_[t] > 0.8:
                            flag = 1
                            suc_counter[e,l] += 1
                            energies[l] = np.mean(group_e[l,t:])
                            pol_at_conv[e,l] = mean_epi_[t]
                            break
                        else:
                            flag = 0
                            pol_at_conv[e,l] = mean_epi_[t]
                            tt = t
                        energies[l] = np.mean(group_e[l,tt:])
                    if flag:
                        pol_grp[l,:] = 1
                    else:
                        unpol_grp[l,:] = 1
                        # energies[l] = np.mean(group_e[l,tt-10:tt])
                    # energies[l] = group_e[l,tt]
                    
                
                # polarised = energies*pol_grp[:,-1]
                # unpolarised = energies*unpol_grp[:,-1]
                
                
                g_conv_vals.append(np.mean(energies))
                g_conv_vals_all.append(energies)
                g_conv_std_vals.append(np.std(energies))

            # ax_v.errorbar((np.round(np.array(sp),decimals=1)*100+fi*1.5),(np.array(g_conv_vals)),yerr=g_conv_std_vals,color=color[fi],label=Labels[fi])
            std_ = np.std(suc_counter,axis=1)
            mean_ = np.mean(suc_counter,axis=1)
            if nC == 1:
                set_axis_leg(ax_v[quad[str(fi)]], Labels[fi],color[fi])
            values = {}
            for i in range(len(sp)):
                if nC == 2:
                    if mean_[i]>0.8:
                        values[str(np.round(sp[i],decimals=1))] = g_conv_vals_all[i][pol_at_conv[i]>0.8].tolist()
                        # print(len(g_conv_vals_all[i][pol_at_conv[i]>0.8]))
                        # parts = ax_v[quad[str(fi)]].violinplot(g_conv_vals_all[i][pol_at_conv[i]>0.8],positions=[sp[i]*100], widths=8,showmeans=True,showmedians=False,showextrema=True)
                        # for pc in parts['bodies']: # type: ignore
                        #     pc.set_facecolor(color[fi])
                        #     pc.set_edgecolor('none')
                        #     pc.set_alpha(0.2)
                        #     # if nC == 0:
                        #     #     pc.set_hatch('////')

                        #     parts['cbars'].set_facecolor(color[fi])
                        #     parts['cbars'].set_edgecolor('#8E006B')
                        #     # parts['cbars'].set_alpha(0.5*(nC+1))
                        #     parts['cmins'].set_facecolor(color[fi])
                        #     parts['cmins'].set_edgecolor('#8E006B')
                        #     # parts['cmins'].set_alpha(0.5*(nC+1))
                        #     parts['cmaxes'].set_facecolor(color[fi])
                        #     parts['cmaxes'].set_edgecolor('#8E006B')
                        #     # parts['cmaxes'].set_alpha(0.5*(nC+1))
                        #     # parts['cmedians'].set_facecolor(color[fi])
                        #     # parts['cmedians'].set_edgecolor('#8E006B')
                        #     # # parts['cmedians'].set_alpha(0.5*(nC+1))
                        #     # parts['cmedians'].set_linestyle('--')
                        #     # parts['cmedians'].set_linewidth(2)
                        #     # parts['cmedians'].set_clim(4)
                        #     parts['cmeans'].set_facecolor(color[fi])
                        #     parts['cmeans'].set_edgecolor('#8E006B')
                        #     # parts['cmeans'].set_alpha(0.5*(nC+1))
                        #     parts['cmeans'].set_linestyle('--')
                        #     parts['cmeans'].set_linewidth(2)
                        # labels = [spn*100 for spn in sp]
                        # set_axis_style(ax_v[quad[str(fi)]], labels,'Time (in sec)',Labels[fi],color[fi])
                        
                # if mean_[i]>0.8:
                #     g_conv_std_vals_max = g_conv_std_vals[i]
                #     if g_conv_std_vals[i] + g_conv_vals[i]>0:
                #         g_conv_std_vals_max = abs(g_conv_vals[i])
                #     if i == 0:
                #         if nC==0:
                #             ax_v[quad[str(fi)]].bar(-10,0,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                #         if fi == 3 and nC==0:
                #             ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='N = '+str(N[0]),width=1)
                #             ax_v[quad[str(fi)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='N = '+str(N[1]),width=1)    
                #     if nC==0:
                #         ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                #     else:
                #         ax_v[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = color[fi],edgecolor=color[fi],width=4,capsize=4,clip_on=True)
                    
                #     if nC==0:
                #         axes[fi].bar(np.round(sp[i]*100+nC*4,decimals=1),g_conv_vals[i],yerr=[[g_conv_std_vals[i]],[g_conv_std_vals_max]],facecolor = 'white',edgecolor=color[fi],width=4,capsize=4,clip_on=True)

            ii += 1
            if nC==2:
                for s in np.round(np.arange(0,1.1,0.1),decimals=1):
                    if str(s) not in values.keys():
                        values[str(s)] = []
                new_vals = {}
                for s in np.round(np.arange(0,1.1,0.1),decimals=1):
                    new_vals[str(s)] = values[str(s)]
                with open(path[:-8]+'/'+str(nAgents)+Labels[fi]+'enrgies_defection.npy', 'w') as f:
                    # Write the dictionary to the file in JSON format
                    json.dump(new_vals, f)
        nC += 1
    # fig_v.supxlabel('Proportion of defectors', fontsize=24,fontweight='bold')
    
    # # fig_v.supylabel('Mean Energy ('+r'$\frac{\Sigma_{t= p>0.8}^{200} e(t)}{\Sigma_{t= p>0.8}^{200}\delta t}$'+')', fontsize=24,fontweight='bold')
    # fig_v.supylabel(r'${\bf \mu_{e(t=p>0.8, t=200)}}$', fontsize=32,fontweight='bold')
    # fig_v.suptitle('Above Cut-off = 0.8', fontsize=24,fontweight='bold')
    # ax_v[0,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    # ax_v[0,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    # ax_v[1,1].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    # ax_v[1,0].legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    # ax_v[quad[str(0)]].indicate_inset_zoom(ax2, edgecolor="black")
    # ax_v[quad[str(1)]].indicate_inset_zoom(ax3, edgecolor="black")
    # ax_v[quad[str(2)]].indicate_inset_zoom(ax4, edgecolor="black")
    # ax_v[quad[str(3)]].indicate_inset_zoom(ax5, edgecolor="black")
    # ax_v[0,0].set_ylim(-60,0)
    # ax_v[0,1].set_ylim(-60,0)
    # ax_v[1,0].set_ylim(-60,0)
    # ax_v[1,1].set_ylim(-60,0)
    # ax2.set_ylim(-2,0) 
    # ax3.set_ylim(-2,0) 
    # ax4.set_ylim(-2,0) 
    # ax5.set_ylim(-2,0) 
    # ax.set_xlabel('% of defectors', fontsize=24)
    # ax.set_ylabel('Time (in sec)', fontsize=24)
    # # ax.set_title('For 98% cohesion convergence', fontsize=24)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    # plt.tight_layout()
    # # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig_v.savefig(path_here+'/50n_energy_val.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

