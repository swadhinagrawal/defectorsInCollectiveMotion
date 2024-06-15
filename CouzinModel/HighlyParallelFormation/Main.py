from multiprocessing import Pool
from functools import partial
# import numpy as np
import swarmGame as sg
import parameters as params
import animation as an
import logger as lg
import numpy as np
import copy as cp
import os
# import matplotlib.pyplot as plt

def resetParams(P, ep, sp):
    # P.setAtt('nAgents', f)
    P.setAtt('Episode', ep)
    P.setAtt('switchProb', sp)
    seed = ep#np.random.randint(0,1000000)
    P.setAtt('seed', seed)
    
    
def Main(P, anim = None, surfix = ''):
    # t1 = time.time()
    # surfix = '_alldlR'
    # surfix = '_50n150szoodlr'
    np.random.seed(P.getAtt('seed'))
    sim = sg.Simulator(P=P)
    
    # Data logging initiator
    Data = []
    accessibility = lg.dataAccess('/DataAnalysis/results'+surfix)
    if 'results'+surfix not in os.listdir(os.path.realpath(os.path.dirname(__file__))+'/DataAnalysis'):
        os.mkdir(os.path.realpath(os.path.dirname(__file__))+'/DataAnalysis/results'+surfix)
    # t2 = time.time()
    # print('\nLoading time: ', t2-t1)
    done = False
    while not done:
        if P.getAtt('render'):
            anim.renderer(sim.interact.agents,P)# type: ignore
            # anim.plotter(sim.interact.agents,P)# type: ignore
        if P.getAtt('logging'):    
            # Data logging
            log = lg.Vault(sim.interact)
            Data.append(cp.deepcopy(log))
        
        for a in sim.interact.agents: # type: ignore
            a.revive()
            
        sim.step(surfix)

        P.setAtt('timer', P.getAtt('timer')+P.getAtt('dt'))
        done = sim.check_reached()
    
    # Data dumping
    if P.getAtt('logging'):
        accessibility.dumper(Data, P.getAtt('switchProb'), P.getAtt('Episode'), '/P_sp_')

def objectWrapper(inp,func):
    if len(inp)==2:
        return getattr(inp[0],func)(val = inp[1])
    elif len(inp)==1:
        return getattr(inp[0],func)()
    
def parallelObject(func,inputs):
    batch_size = len(inputs)
    inps = list(zip(inputs,))
    output = []
    for i in range(0,len(inputs),batch_size):
        opt_var = []
        with Pool(14) as processor:#,ray_address="auto") as p:
            opt_var = processor.starmap(partial(objectWrapper,func=func),inps[i:i+batch_size])

        output += list(opt_var)
    return output

def parallel(func,inputs):
    batch_size = len(inputs)
    inps = list(zip(inputs,))
    output = []
    for i in range(0,len(inputs),batch_size):
        opt_var = []
        with Pool(14) as processor:#,ray_address="auto") as p:
            opt_var = processor.starmap(func,inps[i:i+batch_size])

        output += list(opt_var)
    return output

if __name__=='__main__':
    # followers = [100]
    switchProb = np.round([0.1*i for i in range(11)],decimals = 1)
    # plt.ion()
    P = params.Params()
    # t1 = time.time()
    # surfix = '_alld'
    # surfix = '_zood'
    # surfix = '_zord'
    surfix = '_zoad'
    if 'results'+surfix not in os.listdir(os.path.realpath(os.path.dirname(__file__))+'/DataAnalysis'):
        os.mkdir(os.path.realpath(os.path.dirname(__file__))+'/DataAnalysis/results'+surfix)
    paramsPack = []
    for switch in switchProb:
        # for f in followers:
        for e in range(P.getAtt('nEpisodes')):
            resetParams(P, ep=e, sp = np.round(switch,decimals=2))
            
            paramsPack.append(cp.deepcopy(P))
    # t2 = time.time()
    # print('\nPacking time: ', t2-t1)
            
    if not P.getAtt('render') and P.getAtt('logging'):
        parallel(partial(Main,surfix=surfix),paramsPack)
        # for pack in paramsPack:
        #     Main(pack)
    else:
        for pack in paramsPack:
            anim = an.Animator()
            # t1 = time.time()
            Main(pack, anim=anim)
            # t2 = time.time()
            # print('\nSimulation time: ', t2-t1)