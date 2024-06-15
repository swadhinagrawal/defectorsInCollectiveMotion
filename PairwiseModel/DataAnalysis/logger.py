import pickle as pkl
import copy as cp
import os
import numpy as np

class Vault:
    def __init__(self, world = None):
        print('\nLogging data . . .')
        self.world = world

class dataAccess:
    def __init__(self, loc):
        print('\nData accesser . . .')
        self.path = os.path.realpath(os.path.dirname(__file__)) + loc
    
    def dumper(self, data, f, ep, initials):
        file = open(self.path + initials +str(f)+'_ep_'+str(ep)+'.pkl','wb')
        pkl.dump(data, file)
        file.close()
    
    def loader(self, f, ep, initials):
        file = open(self.path + initials +str(f)+'_ep_'+str(ep)+'.pkl','rb')
        data = pkl.load(file)
        file.close()
        return data
    
    def scalarExtract(self, name, ep, nA, t, f, initials):
        tensor = np.zeros((ep, nA, t))
        for e in range(ep):
            par = self.loader(f = f, ep = e, initials = initials)
            for ti in range(t):
                for a in par[ti].world.agents:
                    tensor[e,a.id,ti] = a.getAttr(name)
        return tensor

    def vectorExtract(self, name, ep, nA, t, f, initials):
        tensor = np.zeros((ep, nA, t, 2))
        for e in range(ep):
            par = self.loader(f = f, ep = e, initials = initials)
            for ti in range(t):
                for a in par[ti].world.agents:
                    for i in range(2):
                        tensor[e,a.id,ti,i] = a.getAttr(name)[i]
        return tensor
    
        
