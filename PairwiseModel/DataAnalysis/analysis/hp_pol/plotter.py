import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import rc
import random
import copy as cp


rc('font', size=16)

path = os.path.realpath(os.path.dirname(__file__)) + '/results'
path_here = os.path.realpath(os.path.dirname(__file__))


n_episodes = 20
max_t = 2001
nAgents = 50

sp = np.round([0.1*i for i in range(11)],decimals=2) # type: ignore

color = ['r','g','b','c','m','y','k','orange','purple','brown','pink']

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

from scipy.signal import savgol_filter
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


plot_convergence_time1 = 0
if plot_convergence_time1:
    number_of_colors = len(sp)
    window = 100
    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #             for i in range(number_of_colors)]
    Labels = ['all zones','ZOA','ZOO','ZOR']
    
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])

    fig,ax = plt.subplots()
    fig_vp,ax_vp = plt.subplots()
    sigma = [0.01, 0.01, 0.01, 0.005]
    ii = 0
    
    
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        conv_time = []
        conv_val = []
        sp_new = []
        for e in range(len(ee)):
            mean_epi = np.sum(np.array(ee[e]),axis=0)/np.array(ee[e]).shape[0]
            for t in range(mean_epi.shape[0],window,-1):
                if np.std(mean_epi[(t-window):(t+1)]) >= np.std(mean_epi[-window:])+sigma[ii]: # type: ignore
                    conv_time.append(np.round((t+int(window/2))/10.0, decimals=1))
                    conv_val.append(np.mean(mean_epi[int(t-window):t+1]))
                    sp_new.append(sp[e])
                    break

        ax.plot(np.round(np.array(sp_new),decimals=2)*100,np.array(conv_time),c=color[ii],label=Labels[ii])
        ax_vp.plot((np.round(np.array(sp_new),decimals=2)*100).astype(int),np.array(conv_val),c=color[ii],label=Labels[ii])
        ii+=1
    ax_vp.set_xlabel('% of defectors', fontsize=18)
    ax_vp.set_ylabel('Polarization', fontsize=18)
    ax.set_xlabel('% of defectors', fontsize=18)
    ax.set_ylabel('Time (in sec)', fontsize=18)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/polarization_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_vp.savefig(path_here+'/polarization_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()


# surfix = '_zoadlR'
# surfix = '_zordlR'
# surfix = '_zoodlR'
# surfix = '_alldlR'

plot_group_count = 0
if plot_group_count:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'clust' in f and '.npy' in f])
    data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])
    
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    Labels = ['All zones','ZOA','ZOO','ZOR']

    for fi in range(len(data_files)):
        ee = Extract_vals(data_files[fi])
        ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])
        i = 0
        for e in range(len(ee)):
            pol_grp = np.zeros(np.array(ee[e]).shape)
            unpol_grp = np.zeros(np.array(ee[e]).shape)
            for l in range(np.array(ee[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi_ = ee_pol[e][l]
                for t in range(len(ee[e][l])):
                    if mean_epi_[t] > 0.8:
                        flag = 1
                        break
                    else:
                        flag = 0
                if flag:
                    pol_grp[l,:] = 1
                else:
                    unpol_grp[l,:] = 1
            
            polarised = ee[e]*pol_grp
            unpolarised = ee[e]*unpol_grp
            
            # Average cluster count at the end of simulation for each fraction of defectors
            mean_epi = np.sum(np.array(ee[e])[:,-1])/np.array(ee[e]).shape[0]
            mean_epi_std = np.std(np.array(ee[e])[:,-1])
            min_mean_epi = np.min(np.array(ee[e])[:,-1])
            max_mean_epi = np.max(np.array(ee[e])[:,-1])
            # ax.bar(sp[i]*100+fi*1.5,mean_epi,color=color[fi],yerr=np.array([[np.clip(mean_epi_std,1,50)], [np.clip(mean_epi_std,0,50)]]),label=Labels[fi])
            if e == 0:
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[min_mean_epi,min_mean_epi],color=color[fi],label=Labels[fi],linestyle='-')
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[max_mean_epi,max_mean_epi],color=color[fi],linestyle='-.')
                ax.plot([sp[i]*100+fi*1.5,sp[i]*100+fi*1.5],[min_mean_epi,max_mean_epi],color=color[fi])
                ax.scatter([sp[i]*100+fi*1.5],[mean_epi],color=color[fi])
                if fi == 0:
                    ax.plot([],[],color='black',linestyle='-.',label='Maximum')
                    ax.plot([],[],color='black',linestyle='-',label='Minimum')
                    ax.scatter([],[],color='black',label=r'$\mu$')
            else:
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[min_mean_epi,min_mean_epi],color=color[fi],linestyle='-')
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[max_mean_epi,max_mean_epi],color=color[fi],linestyle='-.')
                ax.plot([sp[i]*100+fi*1.5,sp[i]*100+fi*1.5],[min_mean_epi,max_mean_epi],color=color[fi])
                ax.scatter([sp[i]*100+fi*1.5],[mean_epi],color=color[fi])
                
            
            # Splitting probability
            mean_epi_pol = np.sum((np.array(polarised[:,-1])>1).astype(int))/len(pol_grp[:,-1])
            
            mean_epi_unpol = np.sum((np.array(unpolarised[:,-1])>1).astype(int))/len(unpol_grp[:,-1])
            if e == 0:
                ax1.bar(sp[i]*100+fi*1.5,mean_epi_pol,color=color[fi],label=Labels[fi],width=1)
                ax2.bar(sp[i]*100+fi*1.5,mean_epi_unpol,color=color[fi],label=Labels[fi],width=1)
            else:
                ax1.bar(sp[i]*100+fi*1.5,mean_epi_pol,color=color[fi],width=1)
                ax2.bar(sp[i]*100+fi*1.5,mean_epi_unpol,color=color[fi],width=1)
            i += 1

    ax.set_xlabel('%D', fontsize=18)
    ax.set_ylabel('Cluster count', fontsize=18)
    ax1.set_xlabel('%D', fontsize=18)
    ax1.set_ylabel('Probability of splitting', fontsize=18)
    ax1.set_title('Polarized')
    ax2.set_xlabel('%D', fontsize=18)
    ax2.set_ylabel('Probability of splitting', fontsize=18)
    ax2.set_title('Unpolarized')
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),bbox_to_anchor=(1.01, 1),ncols=1)
    ax2.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/clust'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/Polsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig2.savefig(path_here+'/Unpolsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_group_count1 = 0
if plot_group_count1:
    data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'clust' in f and '.npy' in f])
    data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])
    
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()

    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    Labels = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$']

    for fi in range(len(data_files)):
        ee = Extract_vals(data_files[fi])
        ee_pol,std_ee = Extract_vals_std(data_files_pol[fi])
        i = 0
        for e in range(len(ee)):
            pol_grp = np.zeros(np.array(ee[e]).shape)
            unpol_grp = np.zeros(np.array(ee[e]).shape)
            for l in range(np.array(ee[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi_ = ee_pol[e][l]
                for t in range(len(ee[e][l])):
                    if mean_epi_[t] > 0.8:
                        flag = 1
                        break
                    else:
                        flag = 0
                if flag:
                    pol_grp[l,:] = 1
                else:
                    unpol_grp[l,:] = 1
            
            polarised = ee[e]*pol_grp
            unpolarised = ee[e]*unpol_grp
            
            # Average cluster count at the end of simulation for each fraction of defectors
            mean_epi = np.sum(np.array(ee[e])[:,-1])/np.array(ee[e]).shape[0]
            mean_epi_std = np.std(np.array(ee[e])[:,-1])
            min_mean_epi = np.min(np.array(ee[e])[:,-1])
            max_mean_epi = np.max(np.array(ee[e])[:,-1])
            # ax.bar(sp[i]*100+fi*1.5,mean_epi,color=color[fi],yerr=np.array([[np.clip(mean_epi_std,1,50)], [np.clip(mean_epi_std,0,50)]]),label=Labels[fi])
            if e == 0:
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[min_mean_epi,min_mean_epi],color=color[fi],label=Labels[fi],linestyle='-')
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[max_mean_epi,max_mean_epi],color=color[fi],linestyle='-.')
                ax.plot([sp[i]*100+fi*1.5,sp[i]*100+fi*1.5],[min_mean_epi,max_mean_epi],color=color[fi])
                ax.scatter([sp[i]*100+fi*1.5],[mean_epi],color=color[fi])
                if fi == 0:
                    ax.plot([],[],color='black',linestyle='-.',label='Maximum')
                    ax.plot([],[],color='black',linestyle='-',label='Minimum')
                    ax.scatter([],[],color='black',label=r'$\mu$')
            else:
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[min_mean_epi,min_mean_epi],color=color[fi],linestyle='-')
                ax.plot([sp[i]*100+fi*1.5-0.5,sp[i]*100+fi*1.5+0.5],[max_mean_epi,max_mean_epi],color=color[fi],linestyle='-.')
                ax.plot([sp[i]*100+fi*1.5,sp[i]*100+fi*1.5],[min_mean_epi,max_mean_epi],color=color[fi])
                ax.scatter([sp[i]*100+fi*1.5],[mean_epi],color=color[fi])
                
            
            # Splitting probability
            mean_epi_pol = np.sum((np.array(polarised[:,-1])>1).astype(int))/len(pol_grp[:,-1])
            
            mean_epi_unpol = np.sum((np.array(unpolarised[:,-1])>1).astype(int))/len(unpol_grp[:,-1])
            if e == 0:
                ax1.bar(np.round(sp[i]*100+fi*1.5,decimals=1),mean_epi_unpol, facecolor = 'white',edgecolor=color[fi],width=1)
                ax1.bar(np.round(sp[i]*100+fi*1.5,decimals=1),bottom=mean_epi_unpol,height=mean_epi_pol,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                if fi == 0:
                    ax1.bar(-10,0, facecolor = 'white',edgecolor= 'black',label='p<=0.8',width=1)
                    ax1.bar(-10,0,facecolor = 'black',edgecolor='black',label='p>0.8',width=1)
            else:
                ax1.bar(np.round(sp[i]*100+fi*1.5,decimals=1),mean_epi_unpol, facecolor = 'white',edgecolor=color[fi],width=1)
                ax1.bar(np.round(sp[i]*100+fi*1.5,decimals=1),bottom=mean_epi_unpol,height=mean_epi_pol,facecolor = color[fi],edgecolor=color[fi],width=1)
            i += 1

    ax.set_xlabel('%D', fontsize=18)
    ax.set_ylabel('Cluster count', fontsize=18)
    ax1.set_xlabel('%D', fontsize=18)
    ax1.set_ylabel('Probability of splitting', fontsize=18)
    ax1.set_ylim(-0.01,1.01)
    ax1.set_xlim(-1,110.01)
    # ax1.set_title('Polarized')
    # ax2.set_xlabel('%D', fontsize=18)
    # ax2.set_ylabel('Probability of splitting', fontsize=18)
    # ax2.set_title('Unpolarized')
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    # ax2.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/clust'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/split'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig2.savefig(path_here+'/Unpolsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

def set_axis_style(ax, labels,ylab,labe,col):
    ax.set_xticks(np.array(labels).astype(int),labels=np.array(labels).astype(int))
    ax.set_xlim(min(labels)-5, max(labels) + 10)
    
    # ax.set_xlabel('Proportion of defectors')
    # ax.set_facecolor('#D4D4D4')
def set_axis_leg(ax, labe,col):    
    ax.plot([],[],color='#8E006B',linestyle='--',label=r'$\mu$')
    ax.plot([],[],color='#8E006B',linestyle=':',label='Median')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1,bbox_to_anchor=(1.2, 1))

plot_polarization_and_misalignment_ep = 1
if plot_polarization_and_misalignment_ep:
    data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f])

    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    fig4,ax4 = plt.subplots()
    fig5,ax5 = plt.subplots(figsize=(10,6))

    number_of_colors = len(sp)

    # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                # for i in range(number_of_colors)]
    Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
    file_counter = 0
    Labels_f = ['D-'+r'$Z_{all}$','D-'+r'$Z_{a}$','D-'+r'$Z_{o}$','D-'+r'$Z_{r}$', 'D-PW']
    colors = plt.cm.turbo(np.linspace(0,1,len(sp)))
    ax.set_title('For each iteration')
    ax1.set_title('For each iteration')
    # ax2.set_title('Iteration average')
    ax3.set_title('Iteration average')
    ax4.set_title('Polarization > 0.8')
    ax5.set_title('Time to achieve p > 0.8',y=1.05,fontdict={'weight':'bold'})
    for fi in data_files:
        ee,std_ee = Extract_vals_std(fi)
        i = 0
        suc_counter = np.zeros((len(sp),ee[0].shape[0]))
        convergence_time = np.zeros((len(sp),ee[0].shape[0]))
        lins = []
        pol_at_conv = np.zeros((len(sp),ee[0].shape[0]))
        for e in range(len(ee)):
            for l in range(np.array(ee[e]).shape[0]):#np.random.randint(0,np.array(ee[e]).shape[0],2):#range(np.array(ee[e]).shape[0])
                mean_epi = ee[e][l]
                std_epi = std_ee[e][l]
                
                for t in range(len(ee[e][l])):
                    if mean_epi[t] > 0.8:
                        suc_counter[e,l] += 1
                        convergence_time[e,l] = np.round(t/10.0, decimals=1)
                        pol_at_conv[e,l] = mean_epi[t]
                        break
                    else:
                        convergence_time[e,l] = np.round(t/10.0, decimals=1)
                        pol_at_conv[e,l] = mean_epi[t]
                        
                    
                if (i == 4 and l == 0) or (i == 0 and l == 0):
                    ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i], label=Labels[i])
                    ax1.plot(range(ee[e].shape[1]),std_epi,c=color[i], label=Labels[i])
                elif (i == 4 and l != 0) or (i == 0 and l != 0):
                    ax.plot(range(ee[e].shape[1]),mean_epi,c=color[i])
                    ax1.plot(range(ee[e].shape[1]),std_epi,c=color[i])
            
            mean_p = np.mean(ee[e],axis = 0)
            std_p = np.std(ee[e],axis = 0)
            ax2.plot([it/10.0 for it in range(std_ee[e].shape[1])],mean_p,color=colors[i],label=Labels[i])

            lins.append(ax2.fill_between([it/10.0 for it in range(std_ee[e].shape[1])],np.clip(mean_p-std_p,0,1),np.clip(mean_p+std_p,0,1),alpha=0.1,color=colors[i]))
            mean_stdp = np.mean(np.array(std_ee[e]),axis=0)
            std_stdp = np.std(np.array(std_ee[e]),axis=0)
            
            ax3.plot(range(std_ee[e].shape[1]),mean_stdp,color=color[i],label=Labels[i])
            ax3.fill_between(range(std_ee[e].shape[1]),mean_stdp-std_stdp,mean_stdp+std_stdp,alpha=0.1,color=color[i])
            i += 1
        c = np.arange(len(sp)+1)
        cmap = plt.get_cmap("turbo", len(c))
        norm = mpl.colors.BoundaryNorm(np.arange(len(c))+0.5,len(c))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cmap.set_array([])

        cbar = fig2.colorbar(cmap,ticks=[1,6,11])
        cbar.ax.set_yticklabels([0,50,100]) 
        cbar.ax.set_ylabel('Proportion of defectors',  fontdict={'weight':'bold','size':18}) 
        # ax4.bar(np.array(sp)*100+file_counter*2,np.mean(suc_counter,axis=1),yerr=np.std(suc_counter,axis=1),label=Labels_f[file_counter], color=color[file_counter],width=1.5)
        ax4.scatter(np.array(sp)*100+file_counter*1.5,np.mean(suc_counter,axis=1),color=color[file_counter])
        std_ = np.std(suc_counter,axis=1)
        mean_ = np.mean(suc_counter,axis=1)
        for i in range(len(sp)):
            ax4.plot([sp[i]*100.0+file_counter*1.5-0.5,sp[i]*100.0+file_counter*1.5+0.5],[np.clip(mean_[i]-std_[i],0,1),np.clip(mean_[i]-std_[i],0,1)],color=color[file_counter],linestyle='-')
            ax4.plot([sp[i]*100.0+file_counter*1.5-0.5,sp[i]*100.0+file_counter*1.5+0.5],[np.clip(std_[i]+mean_[i],0,1),np.clip(std_[i]+mean_[i],0,1)],color=color[file_counter],linestyle='-')
            ax4.plot([sp[i]*100.0+file_counter*1.5,sp[i]*100.0+file_counter*1.5],[np.clip(mean_[i]-std_[i],0,1),np.clip(std_[i]+mean_[i],0,1)],color=color[file_counter])
            
        # ax5.bar(np.array(sp)*100+file_counter*2,np.mean(convergence_time,axis=1),yerr=np.std(convergence_time,axis=1),label=Labels_f[file_counter], edgecolor=color[file_counter], facecolor='white',width=1.5,capsize=3)
        # ax5.errorbar(np.array(sp)*100+file_counter*2,np.mean(convergence_time,axis=1),yerr=np.std(convergence_time,axis=1),label=Labels_f[file_counter], color=color[file_counter],linestyle=':',marker='o')
        set_axis_leg(ax5, None,None) 
        for i in range(len(sp)):    
            if mean_[i]>0.8:
                print(len(convergence_time[i][pol_at_conv[i]>0.8]))
                parts = ax5.violinplot(convergence_time[i][pol_at_conv[i]>0.8],positions=[sp[i]*100], widths=8,showmeans=True,showmedians=True,showextrema=True)
                for pc in parts['bodies']: # type: ignore
                    pc.set_facecolor(color[file_counter])
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.2)
                    # if nC == 0:
                    #     pc.set_hatch('////')

                    parts['cbars'].set_facecolor(color[file_counter])
                    parts['cbars'].set_edgecolor('#8E006B')
                    # parts['cbars'].set_alpha(0.5*(nC+1))
                    parts['cmins'].set_facecolor(color[file_counter])
                    parts['cmins'].set_edgecolor('#8E006B')
                    # parts['cmins'].set_alpha(0.5*(nC+1))
                    parts['cmaxes'].set_facecolor(color[file_counter])
                    parts['cmaxes'].set_edgecolor('#8E006B')
                    # parts['cmaxes'].set_alpha(0.5*(nC+1))
                    parts['cmedians'].set_facecolor(color[file_counter])
                    parts['cmedians'].set_edgecolor('#8E006B')
                    # parts['cmedians'].set_alpha(0.5*(nC+1))
                    parts['cmedians'].set_linestyle('--')
                    parts['cmedians'].set_linewidth(2)
                    parts['cmedians'].set_clim(4)
                    parts['cmeans'].set_facecolor(color[file_counter])
                    parts['cmeans'].set_edgecolor('#8E006B')
                    # parts['cmeans'].set_alpha(0.5*(nC+1))
                    parts['cmeans'].set_linestyle(':')
                    parts['cmeans'].set_linewidth(2)
                labels = [int(spn*100) for spn in sp]
                set_axis_style(ax5, labels,'Time (in sec)',Labels_f[file_counter],color[file_counter])
                                    
        
        file_counter+=1

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel('Group polarization', fontsize=18)
    ax1.set_xlabel(r'$t$', fontsize=18)
    ax1.set_ylabel('Misalignment', fontsize=18)
    ax2.set_xlabel('Time (in sec)',  fontdict={'weight':'bold','size':18})
    ax2.set_ylabel('Group polarization, p',  fontdict={'weight':'bold','size':18})
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([0.0,0.5,1.0])
    ax3.set_xlabel(r'$t$', fontsize=18)
    ax3.set_ylabel('Misalignment', fontsize=18)
    ax4.set_xlabel('%D', fontsize=18)
    ax4.set_ylabel('Formation success rate', fontsize=18)
    ax5.set_xlabel('Proportion of defectors', fontdict={'weight':'bold','size':18})
    ax5.set_ylabel('Time (in sec)',  fontdict={'weight':'bold','size':18})
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    

    ax5.set_ylim(-1,205)
    ax4.set_ylim(-0.05,1.05)
    
    # ax.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax1.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax2.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=1,bbox_to_anchor=(1.1, 1))#bbox_to_anchor=(1.1, 1),
    # ax3.legend(frameon=False,  prop=dict(weight='bold',size=12),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax4.legend(frameon=False,  prop=dict(weight='bold',size=12),bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.1, 1),
    # ax5.legend(frameon=False,  prop=dict(weight='bold',size=12))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/pol'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/misa'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig2.savefig(path_here+'/pol_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig3.savefig(path_here+'/misa_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig4.savefig(path_here+'/FSR'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig5.savefig(path_here+'/t2r0p8'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

