import numpy as np
from muscle import Muscle
from scipy import optimize
from inputs import *
import matplotlib.pyplot as plt

arms_list=[]
muscles_list = []
for i in range(len(name)):
        muscle = Muscle(proximal_attachment[i], distal_attachment[i], alfa[i], 
                        optimal_fiber_length[i], max_iso_force[i], name[i],
                        tendon_length[i],density_MS[i], weight[i])
        muscles_list.append(muscle)
        arms_list.append(muscle.arm())
        arms=np.transpose(np.array(np.squeeze(arms_list)))    
arms[:, -3:] = -0.02
#\\\\\\\\\\\\\\\\\\\\\\\\MUSCLE REDUNDANCY\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#--------------------SIMPLE OPTIMATIZATION OF FORCES OF MUSCLES----------------
def G1(force_1):
    G1 = np.sum([force_1[f]**2 for f in range(len(name))])
    return G1

def eq_c1(force_1):
    return moment_of_elbow_A[i]-np.sum(force_1*arms[i,:])

x0_1 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_1 = [(0, m.get_max_force()) for m in muscles_list]
c_1 = ({'type': 'eq', 'fun':  eq_c1 })
result_G1=[]
for i in range(len(time)):
    result_1=optimize.minimize(G1,x0_1,method='SLSQP',
                              constraints=c_1,bounds=bnds_1,
                              options={'ftol': 1e-5,'maxiter': 15000})
    result_G1.append(result_1.x)   
    # print(result_1.message)
    
#--------------------HILL MODEL------------------------------------------------
def hill_model():
    arm_ext=0.02
    actual_muscle_lengths = []  
    for muscle in muscles_list[:4]:
        a_F=((np.linalg.norm(muscle.distal_attachment,axis=1))**2+(np.linalg.norm(muscle.proximal_attachment,axis=1))**2-
             2*np.linalg.norm(muscle.distal_attachment,axis=1)*np.linalg.norm(muscle.proximal_attachment,axis=1)*np.cos(phi))**0.5
        actual_muscle_lengths.append(a_F)

    for muscle in muscles_list[4:]:
        a_E = ((muscle.proximal_attachment[:,1])**2 - arm_ext**2)**0.5
        c_E= arm_ext*((360*np.pi)/180 - phi - np.arccos(1*np.pi/180) - np.arccos(arm_ext/muscle.proximal_attachment[:,1]))
        length=a_E+c_E
        actual_muscle_lengths.append(length)
    actual_muscle_length=np.transpose(actual_muscle_lengths)
    
    # optimal_fiber = np.mean(actual_muscle_length, axis=0)
    # print(optimal_fiber)
  
    contraction_velocity = np.empty((50,7))
    for i in range(actual_muscle_length.shape[1]):
        for j in range(actual_muscle_length.shape[0]-1):
            if j == 0:  
                velocity = (actual_muscle_length[j+1, i] - actual_muscle_length[j, i]) / dt
            elif j == 49:  
                velocity = (actual_muscle_length[j, i] - actual_muscle_length[j-1, i]) / dt
            else:
                velocity = (actual_muscle_length[j+1, i] - actual_muscle_length[j-1, i]) / (2 * dt)
            contraction_velocity[j, i] = velocity
    contraction_velocity = np.insert(contraction_velocity, 0, np.zeros(7), axis=0)
       
    opt_contraction_velocity=np.transpose(optimal_fiber_length/0.1)
    
    velocity_factor=np.empty((51,7))
    for i in range(actual_muscle_length.shape[1]):
        for j in range(actual_muscle_length.shape[0]):
            if j<26:
                v_f=np.array((opt_contraction_velocity-contraction_velocity[j,i])/
                                      (opt_contraction_velocity-C*contraction_velocity[j,i]))
            else:
                v_f=(2*contraction_velocity-B+contraction_velocity*A/max_iso_force)/(contraction_velocity-B)
            velocity_factor=v_f
    
    fla=np.array(1-(((actual_muscle_length/optimal_fiber_length)-1)*2)**2)
    flp=((actual_muscle_length/optimal_fiber_length)**3)*np.exp(8*(actual_muscle_length/optimal_fiber_length)-12.9)

    return flp, fla, velocity_factor

#---------------OPTIMATIZATION OF ACTIVATION IN HILL MODEL --------------------
def G2(a_2):
    G2 = np.array(np.sum([a_2[f]**2 for f in range(len(name))]))
    return G2

def eq_c2(a_2):
    flp, fla, velocity_factor = hill_model()
    return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
                                        +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
                                                *velocity_factor[i][m]*a_2[m]) for m in range(len(name))])
 
x0_2 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_2 = [(0,1) for m in muscles_list]
c_2 = ({'type': 'eq', 'fun':  eq_c2 })
result_G2=[]
for i in range(len(time)):
    result_2=optimize.minimize(G2,x0_2,method='SLSQP',constraints=c_2,
                                bounds=bnds_2, options={'ftol': 1e-5,
                                                        'maxiter': 1000})
    result_G2.append(result_2.x)  
    # print(result_2.message)
#-------------COMPUTING FORCE AFTER OPT. ACTIVATION IN HILL MODEL--------------
def force_2_hill():
    flp, fla, velocity_factor = hill_model()
    F2_hill = max_iso_force*(velocity_factor*fla*result_G2+flp)*np.cos(alfa)
    return F2_hill
F2_hill = force_2_hill()
