from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import matplotlib


rc('font',size=16)

path = os.path.realpath(os.path.dirname(__file__)) + '/results'
path_here = os.path.realpath(os.path.dirname(__file__))


n_episodes = 100
max_t = 2001
nAgents = 50
# followers = [6,16,40,100]
sp = np.round([0.1*i for i in range(11)],decimals=2) # type: ignore

color = ['r','g','b','c','m','y','k','orange','purple','brown','pink']

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
    ax_v.set_xlabel('% of defectors', fontsize=18)
    ax_v.set_ylabel('Rotational Energy ('+r'$-\omega^{2}$'+')', fontsize=18)
    
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
    # ax_vp.set_xlabel('% of defectors', fontsize=18)
    # ax_vp.set_ylabel('Polarization', fontsize=18)
    
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
    # ax_vs.set_xlabel('% of defectors', fontsize=18)
    # ax_vs.set_ylabel('Distance (in units)', fontsize=18)
    
    ax.set_xlabel('% of defectors', fontsize=18)
    ax.set_ylabel('Time (in sec)', fontsize=18)
    # # ax.set_title('For 98% cohesion convergence', fontsize=18)

    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/energy_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_total_e = 0
if plot_total_e:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$', 'D-PW']
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f])
    data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    sigma = [0.02, 0.4, 0.01, 0.7]
    ii = 0
    for fi in range(len(data_files)):
        ee = Extract_vals(data_files[fi])
        ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])

        g_conv_vals = []
        g_conv_std_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            
            pol_grp = np.zeros(np.array(group_e).shape[0])
            unpol_grp = np.zeros(np.array(group_e).shape[0])
            energies = np.zeros(group_e.shape[0])
            
            
            for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi_ = ee_pol[e][l]
                tt = 0
                for t in range(len(ee_pol[e][l])):
                    if mean_epi_[t] > 0.8:
                        flag = 1
                        tt = t
                        break
                    else:
                        flag = 0
                energies[l] = np.sum(group_e[l,:])*0.1
                if flag:
                    pol_grp[l] = 1
                else:
                    unpol_grp[l] = 1
                    # energies[l] = np.mean(group_e[l,tt-10:tt])
                # energies[l] = group_e[l,tt]
                
            
            polarised = energies*pol_grp
            unpolarised = energies*unpol_grp
            
            
            g_conv_vals.append(np.mean(polarised))
            g_conv_std_vals.append(np.std(polarised))
        for i in range(len(sp)):
            ax_v.errorbar((np.round(np.array(sp)[i],decimals=1)*100+fi*1.5),(np.array(g_conv_vals)[i]),yerr=g_conv_std_vals[i],color=color[fi],label=Labels[fi])
        ii += 1
    ax_v.set_xlabel('Proportion of defectors (in percent, %)', fontsize=18)
    ax_v.set_ylabel('Total Energy ('+r'$\Sigma_{t=0}^{200} e(t)\delta t$'+')', fontsize=18)
    

    # ax.set_xlabel('% of defectors', fontsize=18)
    # ax.set_ylabel('Time (in sec)', fontsize=18)
    # # ax.set_title('For 98% cohesion convergence', fontsize=18)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/total_energy_time.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_total_e_before_pol = 0
if plot_total_e_before_pol:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$', 'D-PW']
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f])
    data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots()
    sigma = [0.02, 0.4, 0.01, 0.7]
    ii = 0
    for fi in range(len(data_files)):
        ee = Extract_vals(data_files[fi])
        ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])

        g_conv_vals = []
        g_conv_std_vals = []
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            
            pol_grp = np.zeros(np.array(group_e).shape[0])
            unpol_grp = np.zeros(np.array(group_e).shape[0])
            energies = np.zeros(group_e.shape[0])
            
            
            for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi_ = ee_pol[e][l]
                tt = 0
                for t in range(len(ee_pol[e][l])):
                    if mean_epi_[t] > 0.8:
                        flag = 1
                        tt = t
                        break
                    else:
                        flag = 0
                energies[l] = np.sum(group_e[l,:tt])*0.1
                if flag:
                    pol_grp[l] = 1
                else:
                    unpol_grp[l] = 1
                    # energies[l] = np.mean(group_e[l,tt-10:tt])
                # energies[l] = group_e[l,tt]
                
            
            polarised = energies*pol_grp
            unpolarised = energies*unpol_grp
            
            
            g_conv_vals.append(np.mean(polarised))
            g_conv_std_vals.append(np.std(polarised))
        for i in range(len(sp)):
            ax_v.bar((np.round(np.array(sp)[i],decimals=1)*100+fi*1.5),(np.array(g_conv_vals)[i]),yerr=g_conv_std_vals[i],color=color[fi],label=Labels[fi],capsize=3,width = 4)
        ii += 1
    ax_v.set_xlabel('Proportion of defectors (in percent, %)', fontsize=18)
    ax_v.set_ylabel('Rotational Energy ('+r'$-\omega^{2}$'+')', fontsize=18)
    

    # ax.set_xlabel('% of defectors', fontsize=18)
    # ax.set_ylabel('Time (in sec)', fontsize=18)
    # # ax.set_title('For 98% cohesion convergence', fontsize=18)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/total_energy_before_pol.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()


def set_axis_style(ax, labels,ylab,labe,col):
    ax.set_xticks(np.array(labels).astype(int),labels=np.array(labels).astype(int))
    ax.set_xlim(min(labels)-10, max(labels) + 10)
    ax.set_ylim(-2.5, 0)
    
    # ax.set_xlabel('Proportion of defectors')
    # ax.set_facecolor('#D4D4D4')
def set_axis_leg(ax, labe,col):    
    ax.plot([],[],color='#8E006B',linestyle='--',label=r'$\mu$')
    ax.plot([],[],color='#8E006B',linestyle=':',label='Median')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1,bbox_to_anchor=(1.2, 1))


plot_convergence_time_at_pol = 1
if plot_convergence_time_at_pol:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$', 'D-PW']
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'energy' in f and '.npy' in f])
    data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])

    # fig,ax = plt.subplots()
    fig_v,ax_v = plt.subplots(figsize=(10,6))
    sigma = [0.02, 0.4, 0.01, 0.7]
    ii = 0
    for fi in range(len(data_files)):
        ee = Extract_vals(data_files[fi])
        ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])

        g_conv_vals = []
        g_conv_vals_no_mean = []
        g_conv_std_vals = []
        succ_prob = []
        pol_at_conv = np.zeros((len(sp),ee[0].shape[0]))
        for e in range(len(ee)):
            group_e = np.sum(ee[e],axis=1)
            
            pol_grp = np.zeros(np.array(group_e).shape[0])
            unpol_grp = np.zeros(np.array(group_e).shape[0])
            energies = np.zeros(group_e.shape[0])
            
            
            for l in range(np.array(ee_pol[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi_ = ee_pol[e][l]
                tt = 0
                for t in range(len(ee_pol[e][l])):
                    if mean_epi_[t] > 0.8:
                        flag = 1
                        tt = t

                        break
                    else:
                        flag = 0
                pol_at_conv[e,l] = mean_epi_[tt]
                energies[l] = np.mean(group_e[l,tt:])
                if flag:
                    pol_grp[l] = 1
                else:
                    unpol_grp[l] = 1
                    # energies[l] = np.mean(group_e[l,tt-10:tt])
                # energies[l] = group_e[l,tt]
                
            
            polarised = energies*pol_grp
            unpolarised = energies*unpol_grp
            
            succ_prob.append(np.mean(pol_grp))
            g_conv_vals.append(np.mean(polarised))
            g_conv_vals_no_mean.append(polarised)
            g_conv_std_vals.append(np.std(polarised))
        set_axis_leg(ax_v, None,None) 
        for i in range(len(sp)):
            
            # print(g_conv_vals_no_mean[i][pol_at_conv[i]>0.8])
            if succ_prob[i]>0.8:
                
                parts = ax_v.violinplot(g_conv_vals_no_mean[i][pol_at_conv[i]>0.8],positions=[sp[i]*100], widths=8,showmeans=True,showmedians=True,showextrema=True)
                for pc in parts['bodies']: # type: ignore
                    pc.set_facecolor(color[0])
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.2)
                    # if nC == 0:
                    #     pc.set_hatch('////')

                    parts['cbars'].set_facecolor(color[0])
                    parts['cbars'].set_edgecolor('#8E006B')
                    # parts['cbars'].set_alpha(0.5*(nC+1))
                    parts['cmins'].set_facecolor(color[0])
                    parts['cmins'].set_edgecolor('#8E006B')
                    # parts['cmins'].set_alpha(0.5*(nC+1))
                    parts['cmaxes'].set_facecolor(color[0])
                    parts['cmaxes'].set_edgecolor('#8E006B')
                    # parts['cmaxes'].set_alpha(0.5*(nC+1))
                    parts['cmedians'].set_facecolor(color[0])
                    parts['cmedians'].set_edgecolor('#8E006B')
                    # parts['cmedians'].set_alpha(0.5*(nC+1))
                    parts['cmedians'].set_linestyle('--')
                    parts['cmedians'].set_linewidth(2)
                    parts['cmedians'].set_clim(4)
                    parts['cmeans'].set_facecolor(color[0])
                    parts['cmeans'].set_edgecolor('#8E006B')
                    # parts['cmeans'].set_alpha(0.5*(nC+1))
                    parts['cmeans'].set_linestyle(':')
                    parts['cmeans'].set_linewidth(2)
                labels = [int(spn*100) for spn in sp]
                set_axis_style(ax_v, labels,'Time (in sec)',None,color[0])
                
                                    
        
            
        #     if succ_prob[i]>0.8:
        #         ax_v.errorbar((np.round(np.array(sp)[i],decimals=1)*100+fi*1.5),(np.array(g_conv_vals)[i]),yerr=g_conv_std_vals[i],color=color[fi],label=Labels[fi],marker='o')
        # ax_v.plot((np.round(np.array(sp)[:-1],decimals=1)*100+fi*1.5),(np.array(g_conv_vals)[:-1]),color=color[fi],linestyle=':')
        ii += 1
    ax_v.set_xlabel('Proportion of defectors', fontdict={'weight':'bold','size':18})
    ax_v.set_ylabel(r'${\bf \mu_{e(t=p>0.8, t=200)}}$', fontdict={'weight':'bold','size':18})


    # ax.set_xlabel('% of defectors', fontsize=18)
    # ax.set_ylabel('Time (in sec)', fontsize=18)
    # # ax.set_title('For 98% cohesion convergence', fontsize=18)

    # ax_v.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/energy_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_v.savefig(path_here+'/after_POL_energy_avg.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

