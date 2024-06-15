from cProfile import label
from hmac import new
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import markers, rc
import random
import matplotlib as mpl
import copy as cp
import json
from scipy.__config__ import show


rc('font',size=24)

path = os.path.realpath(os.path.dirname(__file__)) + '/results'
path_here = os.path.realpath(os.path.dirname(__file__))


n_episodes = 20#100
max_t = 2001
N = [20, 50, 80]

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

from pyparsing import line
from scipy.signal import savgol_filter
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels,ylab,labe,col):
    ax.set_xticks(np.array(labels).astype(int),labels=np.array(labels).astype(int))
    ax.set_xlim(min(labels)-5, max(labels) + 10)
    
    # ax.set_xlabel('Proportion of defectors')
    # ax.set_facecolor('#D4D4D4')
def set_axis_leg(ax, labe,col):    
    ax.plot([],[],color='#8E006B',linestyle='--',label=r'$\bf{\mu}$')
    # ax.plot([],[],color='#8E006B',linestyle=':',label='Median')
    rec = mpl.patches.Rectangle((0,0),0,0,fc=col, edgecolor = 'none', linewidth = 0,label=labe,alpha=0.2)
    ax.add_artist(rec)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)#,bbox_to_anchor=(1.2, 1))
    
    
    # ax.set_ylabel('Time (in sec)')
    

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
    ax_vp.set_xlabel('% of defectors', fontsize=24)
    ax_vp.set_ylabel('Polarization', fontsize=24)
    ax.set_xlabel('% of defectors', fontsize=24)
    ax.set_ylabel('Time (in sec)', fontsize=24)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=24))#bbox_to_anchor=(1.1, 1),
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    fig.savefig(path_here+'/polarization_t2c.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig_vp.savefig(path_here+'/polarization_val.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    plt.show()

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

    ax.set_xlabel('%D', fontsize=24)
    ax.set_ylabel('Cluster count', fontsize=24)
    ax1.set_xlabel('%D', fontsize=24)
    ax1.set_ylabel('Probability of splitting', fontsize=24)
    ax1.set_title('Polarized')
    ax2.set_xlabel('%D', fontsize=24)
    ax2.set_ylabel('Probability of splitting', fontsize=24)
    ax2.set_title('Unpolarized')
    # ax1.set_xticks(np.array(sp)*100)
    ax.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=2)#bbox_to_anchor=(1.1, 1),
    ax1.legend(frameon=False,  prop=dict(weight='bold',size=24),bbox_to_anchor=(1.01, 1),ncols=1)
    ax2.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/clust'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/Polsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig2.savefig(path_here+'/Unpolsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()

plot_group_count1 = 0
if plot_group_count1:
    fig,ax = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(15, 8))
    fig1,ax1 = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    quad = {'0':(0,0),'1':(0,1),'2':(1,0),'3':(1,1)}
    ax2 = ax1[quad[str(1)]].inset_axes([0.4, 0.3, 0.4, 0.4])
    ax3 = ax[quad[str(1)]].inset_axes([0.4, 0.3, 0.4, 0.4])
    ax4 = ax[quad[str(2)]].inset_axes([0.2, 0.4, 0.4, 0.4])
    nC = 0
    hat = ['//','\\\\\\\\']
    # hat = ['..','oo']
    for nAgents in N:
        data_files = np.array([f for f in sorted(os.listdir(path_here)) if 'clust' in f and '.npy' in f and str(nAgents)+'A' in f])
        data_files_pol = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])
        
        

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
                # ee[e][ee[e]==1] = 0
                polarised = ee[e]*pol_grp
                polarised[polarised==0] = 1
                unpolarised = ee[e]*unpol_grp
                unpolarised[unpolarised==0] = 1

                # Average cluster count at the end of simulation for each fraction of defectors
                
                mean_epi = np.sum(np.array(polarised)[:,-1])/np.array(polarised).shape[0]
                mean_epi_std = np.std(np.array(polarised)[:,-1])
                min_mean_epi = np.min(np.array(polarised)[:,-1])
                max_mean_epi = np.max(np.array(polarised)[:,-1])
                
                u_mean_epi = np.sum(np.array(unpolarised)[:,-1])/np.array(unpolarised).shape[0]
                u_mean_epi_std = np.std(np.array(unpolarised)[:,-1])
                u_min_mean_epi = np.min(np.array(unpolarised)[:,-1])
                u_max_mean_epi = np.max(np.array(unpolarised)[:,-1])
                
                if e == 0:
                    if nC==0:
                        ax[quad[str(fi)]].bar(-10,0,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                    if fi == 3 and nC==0:
                        ax[quad[str(fi)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='p>0.8',width=1)
                        ax[quad[str(fi)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='p<=0.8',width=1)
                        ax[quad[str(fi)]].bar(-10,0,facecolor = 'white',edgecolor='black',label='N = '+str(N[0]),width=1,hatch=hat[0])
                        ax[quad[str(fi)]].bar(-10,0,facecolor = 'white',edgecolor='black',label='N = '+str(N[1]),width=1,hatch=hat[1])       
                        ax[quad[str(fi)]].plot([],[],color='purple',linestyle=':',label='p>0.8')
                        ax[quad[str(fi)]].plot([],[],color='purple',linestyle='-',label='p<=0.8')

                ax[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),u_mean_epi,facecolor = color[fi],edgecolor='black',width=4,hatch=hat[nC])
                
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_min_mean_epi,u_min_mean_epi],color='purple',linestyle='-')
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_max_mean_epi,u_max_mean_epi],color='purple',linestyle='-')
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_min_mean_epi,u_max_mean_epi],color='purple',linestyle='-')

                ax[quad[str(fi)]].bar(np.round(sp[i]*100+nC*4,decimals=1),bottom=u_mean_epi,height=mean_epi,color = 'none',edgecolor='black',width=4,hatch=hat[nC])
                
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+min_mean_epi,u_mean_epi+min_mean_epi],color='purple',linestyle=':')
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+max_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                ax[quad[str(fi)]].plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_mean_epi+min_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                if fi == 1:
                    # if u_mean_epi>1:
                    ax3.bar(np.round(sp[i]*100+nC*4,decimals=1),u_mean_epi, facecolor = color[fi],edgecolor='black',width=4,hatch=hat[nC])
                    ax3.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_min_mean_epi,u_min_mean_epi],color='purple',linestyle='-')
                    ax3.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_max_mean_epi,u_max_mean_epi],color='purple',linestyle='-')
                    ax3.plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_min_mean_epi,u_max_mean_epi],color='purple',linestyle='-')
                    # if mean_epi>1:
                    ax3.bar(np.round(sp[i]*100+nC*4,decimals=1),bottom=u_mean_epi,height=mean_epi,color = 'none',edgecolor='black',width=4,hatch=hat[nC])
                    ax3.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+min_mean_epi,u_mean_epi+min_mean_epi],color='purple',linestyle=':')
                    ax3.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+max_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                    ax3.plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_mean_epi+min_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                if fi == 2:
                    # if u_mean_epi>1:
                    ax4.bar(np.round(sp[i]*100+nC*4,decimals=1),u_mean_epi, facecolor = color[fi],edgecolor='black',width=4,hatch=hat[nC])
                    ax4.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_min_mean_epi,u_min_mean_epi],color='purple',linestyle='-')
                    ax4.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_max_mean_epi,u_max_mean_epi],color='purple',linestyle='-')
                    ax4.plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_min_mean_epi,u_max_mean_epi],color='purple',linestyle='-')
                    # if mean_epi>1:
                    ax4.bar(np.round(sp[i]*100+nC*4,decimals=1),bottom=u_mean_epi,height=mean_epi,color = 'none',edgecolor='black',width=4,hatch=hat[nC])
                    ax4.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+min_mean_epi,u_mean_epi+min_mean_epi],color='purple',linestyle=':')
                    ax4.plot([sp[i]*100+nC*4-2,sp[i]*100+nC*4+2],[u_mean_epi+max_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                    ax4.plot([sp[i]*100+nC*4,sp[i]*100+nC*4],[u_mean_epi+min_mean_epi,u_mean_epi+max_mean_epi],color='purple',linestyle=':')
                        
                
                # Splitting probability
                mean_epi_pol = np.sum((np.array(polarised[:,-1])>1).astype(int))/len(pol_grp[:,-1])
                
                mean_epi_unpol = np.sum((np.array(unpolarised[:,-1])>1).astype(int))/len(unpol_grp[:,-1])
                if e == 0:
                    if nC==0:
                        ax1[quad[str(fi)]].bar(-10,0,facecolor = color[fi],edgecolor=color[fi],label=Labels[fi],width=1)
                    if fi == 3 and nC==0:
                        ax1[quad[str(fi)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='p>0.8',width=1)
                        ax1[quad[str(fi)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='p<=0.8',width=1)
                        ax1[quad[str(fi)]].bar(-10,0,facecolor = 'white',edgecolor='black',label='N = '+str(N[0]),width=1,hatch=hat[0])
                        ax1[quad[str(fi)]].bar(-10,0,facecolor = 'white',edgecolor='black',label='N = '+str(N[1]),width=1,hatch=hat[1])       

                ax1[quad[str(fi)]].bar(np.round(sp[i]*100+nC*2,decimals=1),mean_epi_unpol, facecolor = color[fi],edgecolor='black',width=2,hatch=hat[nC])
                ax1[quad[str(fi)]].bar(np.round(sp[i]*100+nC*2,decimals=1),bottom=mean_epi_unpol,height=mean_epi_pol,color = 'none',edgecolor='black',width=2,hatch=hat[nC])
                if fi == 1:
                    ax2.bar(np.round(sp[i]*100+nC*2,decimals=1),mean_epi_unpol, facecolor = color[fi],edgecolor='black',width=2,hatch=hat[nC])
                    ax2.bar(np.round(sp[i]*100+nC*2,decimals=1),bottom=mean_epi_unpol,height=mean_epi_pol,color = 'none',edgecolor='black',width=2,hatch=hat[nC])
                        

                i += 1
        nC += 1

    fig.supxlabel('Fraction of defectors', fontsize=24)
    fig.supylabel('Cluster count', fontsize=24)
    fig1.supxlabel('Fraction of defectors', fontsize=24)
    fig1.supylabel('Probability of splitting', fontsize=24)
    ax1[0,0].set_ylim(-0.01,1.01)
    ax1[0,0].set_xlim(-1,110.01)
    ax1[0,1].set_ylim(-0.01,1.01)
    ax1[0,1].set_xlim(-1,110.01)
    ax1[1,0].set_ylim(-0.01,1.01)
    ax1[1,0].set_xlim(-1,110.01)
    ax1[1,1].set_ylim(-0.01,1.01)
    ax1[1,1].set_xlim(-1,110.01)
    ax2.set_xlim(80,110) 
    ax3.set_xlim(87,110) 
    ax4.set_xlim(27,38) 
    ax4.set_ylim(0,5) 
    ax2.set_ylim(0.0,0.02)
    ax1[quad[str(1)]].indicate_inset_zoom(ax2, edgecolor="black")
    ax[quad[str(1)]].indicate_inset_zoom(ax3, edgecolor="black")
    ax[quad[str(2)]].indicate_inset_zoom(ax4, edgecolor="black")
    # ax1.set_title('Polarized')
    # ax2.set_xlabel('%D', fontsize=24)
    # ax2.set_ylabel('Probability of splitting', fontsize=24)
    # ax2.set_title('Unpolarized')
    # ax1.set_xticks(np.array(sp)*100)
    ax[0,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax[0,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax[1,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax[1,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    
    ax1[0,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax1[0,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax1[1,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax1[1,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    # fig1.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1,bbox_to_anchor=(1.3, 0.95))
    # ax2.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    
    plt.tight_layout()
    fig.savefig(path_here+'/clust'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/split'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig2.savefig(path_here+'/Unpolsplit'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.savefig(path_here+'/ro40_v180.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()


# Final figures below

plot_polarization_and_misalignment_ep = 0
if plot_polarization_and_misalignment_ep:
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    fig4,ax4 = plt.subplots()
    fig5,ax5 = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(8,6))
    quad = {'0':(0,0),'1':(0,1),'2':(1,0),'3':(1,1)}
    ax__ = ax5[quad[str(0)]].inset_axes([0.1, 0.4, 0.4, 0.4])
    lineS = [':','-.','--']
    pointS = ['*','x','o']
    nC = 0
    for nAgents in N:
        ax4.plot([],[],color='black',marker=pointS[nC],label='N = '+str(N[nC]))
        # ax4.plot([],[],color='black',linestyle=lineS[nC],label='n = '+str(N[nC]))
        data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])

        number_of_colors = len(sp)

        # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    # for i in range(number_of_colors)]
        Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
        file_counter = 0
        Labels_f = ['D-'+r'${\bf Z_{all}}$','D-'+r'${\bf Z_{a}}$','D-'+r'${\bf Z_{o}}$','D-'+r'${\bf Z_{r}}$']
        
        ax.set_title('For each iteration')
        ax1.set_title('For each iteration')
        ax2.set_title('Iteration average')
        ax3.set_title('Iteration average')
        # ax4.set_title('p > 0.8')
        fig5.suptitle('p > 0.8 and Cut-off = 0.8')
        for fi in data_files:
            ee,std_ee = Extract_vals_std(fi)
            i = 0
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
                        else:
                            convergence_time[e,l] = 200#np.round(t/10.0, decimals=1)
                            
                        
                    if (i == 4 and l == 0) or (i == 0 and l == 0):
                        ax.plot(range(ee[e].shape[1]),mean_epi,c=colorD[i], label=Labels[i])
                        ax1.plot(range(ee[e].shape[1]),std_epi,c=colorD[i], label=Labels[i])
                    elif (i == 4 and l != 0) or (i == 0 and l != 0):
                        ax.plot(range(ee[e].shape[1]),mean_epi,c=colorD[i])
                        ax1.plot(range(ee[e].shape[1]),std_epi,c=colorD[i])
                
                mean_p = np.mean(ee[e],axis = 0)
                std_p = np.std(ee[e],axis = 0)
                ax2.plot(range(std_ee[e].shape[1]),mean_p,color=colorD[i],label=Labels[i])
                ax2.fill_between(range(std_ee[e].shape[1]),mean_p-std_p,mean_p+std_p,alpha=0.1,color=colorD[i])
                mean_stdp = np.mean(np.array(std_ee[e]),axis=0)
                std_stdp = np.std(np.array(std_ee[e]),axis=0)
                
                ax3.plot(range(std_ee[e].shape[1]),mean_stdp,color=colorD[i],label=Labels[i])
                ax3.fill_between(range(std_ee[e].shape[1]),mean_stdp-std_stdp,mean_stdp+std_stdp,alpha=0.1,color=colorD[i])
                i += 1
            # ax4.bar(np.array(sp)*100+file_counter*2,np.mean(suc_counter,axis=1),yerr=np.std(suc_counter,axis=1),label=Labels_f[file_counter], color=color[file_counter],width=1.5)
            
            std_ = np.std(suc_counter,axis=1)
            mean_ = np.mean(suc_counter,axis=1)
            
            if nC == 0 and file_counter ==0:
                ax4.plot([-10,110],[0.8,0.8],color='black',label='Cut-off',linestyle='--')
            if nC==0:
                ax4.plot([],[],color=color[file_counter],label=Labels_f[file_counter],linestyle=':',linewidth=4)
            
              
            
            ax4.plot(np.array(sp)*100+1.5*file_counter,mean_,color=color[file_counter],marker=pointS[nC],markersize=10, linestyle=':') # type: ignore
            mu_t = np.mean(convergence_time,axis=1)
            std_t = np.std(convergence_time,axis=1)
            min_t = np.min(convergence_time,axis=1)
            max_tt = np.max(convergence_time,axis=1)
            
            for i in range(len(sp)):    
                
                if mean_[i]>0.8:
                    # print(mu_t[i],min_t[i],max_tt[i])
                    if max_tt[i]+mu_t[i]>200:
                        max_tt[i] = max_tt[i]-mu_t[i]
                    if i == 0:
                        if nC==0:
                            ax5[quad[str(file_counter)]].bar(-10,0,facecolor = color[file_counter],edgecolor=color[file_counter],label=Labels_f[file_counter],width=1)
                        if file_counter == 3 and nC==0:
                            ax5[quad[str(file_counter)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='N = '+str(N[0]),width=1)
                            ax5[quad[str(file_counter)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='N = '+str(N[1]),width=1)    
                    if nC==0:
                        ax5[quad[str(file_counter)]].bar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[min_t[i]],[max_tt[i]]],facecolor = 'white',edgecolor=color[file_counter],width=4,capsize=4)
                        if file_counter==0 and i<6:
                            ax__.bar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],facecolor = 'white',edgecolor=color[file_counter],width=4)
                    else:
                        ax5[quad[str(file_counter)]].bar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[min_t[i]],[max_tt[i]]],facecolor = color[file_counter],edgecolor=color[file_counter],width=4,capsize=4)
                        if file_counter==0 and i<6:
                            ax__.bar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],facecolor = color[file_counter],edgecolor=color[file_counter],width=4)

                    # if i == 0:
                    #     ax5.bar(np.array(sp)[i]*100+file_counter*2,np.mean(convergence_time,axis=1)[i],yerr=np.std(convergence_time,axis=1)[i],label=Labels_f[file_counter], edgecolor=color[file_counter], facecolor='white',width=1.5,capsize=3)
                    # else:
                    #     ax5.bar(np.array(sp)[i]*100+file_counter*2,np.mean(convergence_time,axis=1)[i],yerr=np.std(convergence_time,axis=1)[i], edgecolor=color[file_counter], facecolor='white',width=1.5,capsize=3)
            
            
            with open(path[:-8]+'/'+str(nAgents)+Labels_f[file_counter]+'FSR.npy', 'wb') as f:
                np.save(f,suc_counter)   
            file_counter+=1
        nC+=1 

    # ax.set_xlabel(r'$t$', fontsize=24)
    # ax.set_ylabel('Group polarization', fontsize=24)
    # ax1.set_xlabel(r'$t$', fontsize=24)
    # ax1.set_ylabel('Misalignment', fontsize=24)
    # ax2.set_xlabel(r'$t$', fontsize=24)
    # ax2.set_ylabel('Group polarization', fontsize=24)
    # ax3.set_xlabel(r'$t$', fontsize=24)
    # ax3.set_ylabel('Misalignment', fontsize=24)
    # ax4.set_xlabel('Proportion of defectors', fontdict={'weight':'bold','size':24})
    # ax4.set_ylabel('Probability of p > 0.8', fontdict={'weight':'bold','size':24})
    # ax4.spines['top'].set_visible(False)
    # ax4.spines['right'].set_visible(False)
    
    # ax5[quad[str(0)]].indicate_inset_zoom(ax__, edgecolor="black")
    # ax__.set_ylim(0,20)
    # # ax5.set_xlabel('%D', fontsize=24)
    # # ax5.set_ylabel('Time (in sec)', fontsize=24)
    # # ax5.set_ylim(0,101)
    # fig5.supxlabel('Proportion of defectors', fontdict={'weight':'bold','size':24})
    # fig5.supylabel('Time (in sec)', fontdict={'weight':'bold','size':24})
    # ax4.set_ylim(-0.05,1.05)
    # ax4.set_xlim(-0.05,110.0)
    
    
    # ax.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax1.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax2.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax3.legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=2)#bbox_to_anchor=(1.1, 1),
    # ax4.legend(frameon=False,  prop=dict(weight='bold',size=24),bbox_to_anchor=(1.01, 1))#bbox_to_anchor=(1.1, 1),
    # # ax5.legend(frameon=False,  prop=dict(weight='bold',size=24))#bbox_to_anchor=(1.1, 1),
    # ax5[0,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    # ax5[0,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    # ax5[1,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    # ax5[1,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    
    # # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    # plt.tight_layout()
    # # fig.savefig(path_here+'/pol'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # # fig1.savefig(path_here+'/misa'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # # fig2.savefig(path_here+'/pol_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # # fig3.savefig(path_here+'/misa_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig4.savefig(path_here+'/FSR'+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    # fig5.savefig(path_here+'/t2r0p8'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # plt.show()

plot_polarization_time = 1
if plot_polarization_time:

    fig5,ax5 = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=(10,6))
    fig,ax = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(10,10))
    fig.supylabel('Time (in sec)', fontsize=24,fontweight='bold')
    fig.supxlabel('Proportion of defectors (in %)', fontsize=24,fontweight='bold')
    fig1,ax1 = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(10,10))
    fig1.supylabel('p', fontsize=24,fontweight='bold')
    fig1.supxlabel('Proportion of defectors (in %)', fontsize=24,fontweight='bold')
    quad = {'0':0,'1':1,'2':2,'3':3}
    nC = 0

    for nAgents in N:

        data_files = np.array([f for f in sorted(os.listdir(path[:-8])) if 'polarization' in f and '.npy' in f and str(nAgents)+'A' in f])

        number_of_colors = len(sp)

        # color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    # for i in range(number_of_colors)]
        Labels = ['%D = '+str(int(np.round(i,decimals=2)*100)) for i in sp]
        file_counter = 0
        Labels_f = ['D-'+r'${\bf Z_{all}}$','D-'+r'${\bf Z_{a}}$','D-'+r'${\bf Z_{o}}$','D-'+r'${\bf Z_{r}}$']
        

        # ax4.set_title('p > 0.8')
        fig5.suptitle('p > 0.8, Cut-off = 0.8' ,  fontsize=24,fontweight='bold')
        for fi in data_files:
            ee,std_ee = Extract_vals_std(fi)
            i = 0
            suc_counter = np.zeros((len(sp),ee[0].shape[0]))
            convergence_time = np.zeros((len(sp),ee[0].shape[0]))
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
                            # convergence_time[e,l] = 200#np.round(t/10.0, decimals=1)
                            convergence_time[e,l] = 200#np.round(t/10.0, decimals=1)
                            pol_at_conv[e,l] = mean_epi[t]

                            
                        

                # if file_counter == 0:
                #     if nC==1 and e ==0:
                #         ax.hist(convergence_time[i],bins=200,label=Labels[e]+Labels_f[file_counter],color=color[file_counter])

                mean_p = np.mean(ee[e],axis = 0)
                std_p = np.std(ee[e],axis = 0)
                mean_stdp = np.mean(np.array(std_ee[e]),axis=0)
                std_stdp = np.std(np.array(std_ee[e]),axis=0)

                i += 1
            # ax4.bar(np.array(sp)*100+file_counter*2,np.mean(suc_counter,axis=1),yerr=np.std(suc_counter,axis=1),label=Labels_f[file_counter], color=color[file_counter],width=1.5)
            
            std_ = np.std(suc_counter,axis=1)
            mean_ = np.mean(suc_counter,axis=1)

            mu_t = np.mean(convergence_time,axis=1)
            std_t = np.std(convergence_time,axis=1)
            min_t = np.min(convergence_time,axis=1)
            max_tt = np.max(convergence_time,axis=1)
            
            ender = 0
            markers=['x','*']
            if nC == 1:
                set_axis_leg(ax[quad[str(file_counter)]], Labels_f[file_counter],color[file_counter])
                # set_axis_leg(ax1[quad[str(file_counter)]], Labels_f[file_counter],color[file_counter])
            values = {}
            for i in range(len(sp)):
                if nC == 2:
                    
                    if mean_[i]>0.8:
                        values[str(np.round(sp[i],decimals=1))] = convergence_time[i][pol_at_conv[i]>0.8].tolist()
                        print(len(convergence_time[i][pol_at_conv[i]>0.8]))
                        parts = ax[quad[str(file_counter)]].violinplot(convergence_time[i][pol_at_conv[i]>0.8],positions=[sp[i]*100], widths=8,showmeans=True,showmedians=False,showextrema=True)
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
                            # parts['cmedians'].set_facecolor(color[file_counter])
                            # parts['cmedians'].set_edgecolor('#8E006B')
                            # # parts['cmedians'].set_alpha(0.5*(nC+1))
                            # parts['cmedians'].set_linestyle('--')
                            # parts['cmedians'].set_linewidth(2)
                            # parts['cmedians'].set_clim(4)
                            parts['cmeans'].set_facecolor(color[file_counter])
                            parts['cmeans'].set_edgecolor('#8E006B')
                            # parts['cmeans'].set_alpha(0.5*(nC+1))
                            parts['cmeans'].set_linestyle('--')
                            parts['cmeans'].set_linewidth(2)
                        labels = [spn*100 for spn in sp]
                        set_axis_style(ax[quad[str(file_counter)]], labels,'Time (in sec)',Labels_f[file_counter],color[file_counter])
                        
                        parts = ax1[quad[str(file_counter)]].violinplot(pol_at_conv[i][pol_at_conv[i]>0.8],positions=[sp[i]*100+nC*5], widths=8,showmeans=True,showmedians=False,showextrema=True)
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
                            # parts['cmedians'].set_facecolor(color[file_counter])
                            # parts['cmedians'].set_edgecolor('#8E006B')
                            # # parts['cmedians'].set_alpha(0.5*(nC+1))
                            # parts['cmedians'].set_linestyle('--')
                            # parts['cmedians'].set_linewidth(2)
                            # parts['cmedians'].set_clim(4)
                            parts['cmeans'].set_facecolor(color[file_counter])
                            parts['cmeans'].set_edgecolor('#8E006B')
                            # parts['cmeans'].set_alpha(0.5*(nC+1))
                            parts['cmeans'].set_linestyle('--')
                            parts['cmeans'].set_linewidth(2)

                        # set style for the axes
                        labels = [spn*100 for spn in sp]
                        set_axis_style(ax1[quad[str(file_counter)]], labels,'p',Labels_f[file_counter],color[file_counter])
            if nC==2:
                # print(values.keys())
                
                for s in np.round(np.arange(0,1.1,0.1),decimals=1):
                    if str(s) not in values.keys():
                        values[str(s)] = []
                new_vals = {}
                for s in np.round(np.arange(0,1.1,0.1),decimals=1):
                    new_vals[str(s)] = values[str(s)]
                with open(path[:-8]+'/'+str(nAgents)+Labels_f[file_counter]+'conv_times.npy', 'w') as f:
                    # Write the dictionary to the file in JSON format
                    json.dump(new_vals, f)  

                if mean_[i]>0.8:
                    # print(mu_t[i],min_t[i],max_tt[i])
                    if max_tt[i] == 200:
                        max_tt[i] = 200-mu_t[i]
                    if mu_t[i]-std_t[i]<0:
                        std_t[i] = 0
                    if mu_t[i]+std_t[i]>200:
                        std_t[i] = 200
                    if i == 0:
                        # if nC==0:
                        #     ax5[quad[str(file_counter)]].bar(-10,0,facecolor = color[file_counter],edgecolor=color[file_counter],label=Labels_f[file_counter],width=1)
                        if file_counter == 3:
                            # ax5[quad[str(file_counter)]].bar(-10,0, facecolor = 'white',edgecolor= 'black',label='N = '+str(N[0]),width=1)
                            # ax5[quad[str(file_counter)]].bar(-10,0, facecolor = 'black',edgecolor= 'black',label='N = '+str(N[1]),width=1)    
                            ax5[quad[str(file_counter)]].plot([],[],color='black',linestyle=':',marker=markers[nC],label='N = '+str(N[nC]))
                    # if nC==0:
                    #     ax5[quad[str(file_counter)]].errorbar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[np.clip(mu_t[i]-std_t[i],0,200)],[np.clip(mu_t[i]+std_t[i],0,200)]],color=color[file_counter],marker='x',markersize = 10)
                        
                        
                    # else:
                    #     ax5[quad[str(file_counter)]].errorbar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[np.clip(mu_t[i]-std_t[i],0,200)],[np.clip(mu_t[i]+std_t[i],0,200)]],color=color[file_counter],marker='*',markersize = 10)
                    # if nC==0:
                    #     ax5[quad[str(file_counter)]].errorbar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[min_t[i]],[max_tt[i]]],color=color[file_counter],marker='x',markersize = 10)
                        
                    # else:
                    #     ax5[quad[str(file_counter)]].errorbar(np.round(sp[i]*100+nC*4,decimals=1),mu_t[i],yerr=[[min_t[i]],[max_tt[i]]],color=color[file_counter],marker='*',markersize = 10)
                    ender = i

            # ax5[quad[str(file_counter)]].plot(np.round(sp[:ender+1]*100+nC*4,decimals=1),mu_t[:ender+1],color=color[file_counter],linestyle=':')
                
            file_counter+=1
        nC+=1 



    # ax5.set_xlabel('%D', fontsize=24)
    # ax5.set_ylabel('Time (in sec)', fontsize=24)
    # ax5.set_ylim(0,101)
    
    fig5.supxlabel('Proportion of defectors', fontsize=24,fontweight='bold')
    fig5.supylabel('Time (in sec)', fontsize=24,fontweight='bold')

    
    ax5[0,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax5[0,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax5[1,1].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax5[1,0].legend(frameon=False,  prop=dict(weight='bold',size=24),ncols=1)
    ax5[0,0].spines['top'].set_visible(False)
    ax5[0,0].spines['right'].set_visible(False)
    ax5[0,1].spines['top'].set_visible(False)
    ax5[0,1].spines['right'].set_visible(False)
    ax5[1,0].spines['top'].set_visible(False)
    ax5[1,0].spines['right'].set_visible(False)
    ax5[1,1].spines['top'].set_visible(False)
    ax5[1,1].spines['right'].set_visible(False)
    
    
    # plt.subplots_adjust(wspace=0.9,left=0.11,right=0.824)
    plt.tight_layout()
    # fig.savefig(path_here+'/pol'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig1.savefig(path_here+'/misa'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig2.savefig(path_here+'/pol_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    # fig3.savefig(path_here+'/misa_std'+'.png',format = "png",bbox_inches="tight",pad_inches=0.2)
    fig5.savefig(path_here+'/t2r0p8'+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    fig.savefig(path_here+'/50n_t2r0p8'+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    fig1.savefig(path_here+'/50n_p'+'.pdf',format = "pdf",bbox_inches="tight",pad_inches=0.2)
    plt.show()
