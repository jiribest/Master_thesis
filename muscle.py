
import numpy as np
from inputs import *


class Muscle:
     def __init__(self, proximal_attachment, distal_attachment, alfa,
                  optimal_fiber_length , max_iso_force, name, tendom_length,
                  density_MS, weight):
        self.distal_attachment = np.array(np.transpose((np.tile(distal_attachment,(len(time),1)))@trans_matrix)[:,0,:])
        self.proximal_attachment = np.tile(proximal_attachment,(len(time),1))
        self.alfa = alfa
        self.optimal_fiber_length = np.tile(optimal_fiber_length,(len(time)))
        self.max_iso_force = max_iso_force
        self.name = name
        self.tendom_length = np.tile(tendom_length,(len(time)))
        self.density_MS = np.tile(density_MS,(len(time)))
        self.weight = np.tile(weight,(len(time)))
        
     def arm(self):
         vector = np.array(self.distal_attachment-self.proximal_attachment)
         vector_norm = np.linalg.norm(vector,axis=1)
         unit_vector=vector/vector_norm[:,np.newaxis]
         get_arm=(np.cross(unit_vector,self.distal_attachment))
         return get_arm
     
     def get_max_force(self):
         return self.max_iso_force