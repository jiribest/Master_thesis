import numpy as np
from muscle import Muscle
from scipy import optimize
from inputs import *
from base_computing_system import hill_model

import matplotlib.pyplot as plt


arms_list=[]
muscles_list = []
for i in range(len(name)):
        muscle = Muscle(proximal_attachment[i], distal_attachment[i], alfa[i], 
                        optimal_fiber_length[i], max_iso_force[i], name[i],
                        tendon_length[i],density_MS[i], weight[i])
        muscles_list.append(muscle)
        arms_list.append(muscle.arm())
        arms=(np.transpose(np.squeeze(arms_list)))      
arms[:, -3:] = -0.02
#/////////////////////          GTO MODELING             //////////////////////
    
hustota = np.transpose(np.array([muscle.density_MS for muscle in muscles_list]))
mi=np.transpose(np.array([muscle.weight for muscle in muscles_list]))


#/////////////////////     LINEAR MODEL         ///////////////////////////////

#---------    OPTIMATIZATION OF FORCES OF MUSCLES through density   -----------
def G3(force_2): 
    G3 = np.sum([(force_2[k])*hustota[0,k]*0.8*mi[0,k] for k in range(7)])
    return G3

def eq_c3(force_2):
    return  moment_of_elbow_A[i] - np.sum(force_2*arms[i, :])
 
x0 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_3 = [(0, m.get_max_force()) for m in muscles_list]
c_3= ({'type': 'eq', 'fun':  eq_c3 })
result_G3=[]
for i in range(len(time)):
    result_3=optimize.minimize(G3,x0,method='SLSQP',
                              constraints=c_3,bounds=bnds_3,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G3.append(result_3.x)  
    #print(result_3.message)
result_G3 = np.array(result_G3)



#--------    OPTIMATIZATION OF activation  through density  -----------------
def G4(a_4):
    flp, fla, velocity_factor = hill_model()
    G4 = np.sum([hustota[0,k]*0.8*mi[0,k]*(max_iso_force[k]*fla[i,k]*velocity_factor[i,k]*(a_4[k])*np.cos(alfa[k])+
                                        max_iso_force[k]*np.cos(alfa[k])*flp[i,k]) for k in range(7)])
    return G4

def eq_c4(a_4):
    flp, fla, velocity_factor = hill_model()
    return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
                                        +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
                                                *velocity_factor[i][m]*a_4[m]) for m in range(len(name))])

x0 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_4 = [(0,1) for m in muscles_list]
c_4 = ({'type': 'eq', 'fun':  eq_c4 })
result_G4=[]
for i in range(len(time)):
    result_4=optimize.minimize(G4,x0,method='SLSQP',
                              constraints=c_4,bounds=bnds_4,
                              options={'ftol': 1e-3,'maxiter': 15000})  #/////////////   HAZI CHYBU PRI VETSI TOLERANCI
    result_G4.append(result_4.x)
    #print(result_4.message)
result_G4 = np.array(result_G4)

#-------------COMPUTING FORCE AFTER OPT. ACTIVATION IN HILL MODEL--------------
def force_4_hill():
    flp, fla, velocity_factor = hill_model()
    F4_hill = max_iso_force*(velocity_factor*fla*result_G4+flp)*np.cos(alfa)
    return F4_hill
F4_hill = force_4_hill()

# /////////////////////    NON-LINEAR MODEL     ///////////////////////////////

#---------    OPTIMATIZATION OF FORCES  through density   ---------------------
def G5(force_3):
    G5 = np.sum([(A_GTO*np.log((force_3[k]/(hustota[0,k]*0.8*mi[0,k]))/B_GTO + 1)*hustota[0,k]*0.8*mi[0,k]) for k in range(7)])
    return G5

def eq_c5(force_3):
    return moment_of_elbow_A[i]-np.sum(force_3*arms[i,:])

x0 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_5 = [(0, m.get_max_force()) for m in muscles_list]
c_5 = ({'type': 'eq', 'fun':  eq_c5 })
result_G5=[]
for i in range(len(time)):
    result_5=optimize.minimize(G5,x0,method='SLSQP',
                              constraints=c_5,bounds=bnds_5,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G5.append(result_5.x)
    #print(result_5.message)
result_G5 = np.array(result_G5)


# # #---------    OPTIMATIZATION OF activation through density   HILL/-----------
def G6(a_6):
    flp, fla, velocity_factor = hill_model()
    G6 = np.sum([(A_GTO*np.log(((max_iso_force[k]*fla[i,k]*velocity_factor[i,k]*(a_6[k])*np.cos(alfa[k])+
                                        max_iso_force[k]*np.cos(alfa[k])*flp[i,k])/(hustota[0,k]*0.8*mi[0,k]))/B_GTO + 1)*hustota[0,k]*0.8*mi[0,k]) for k in range(7)])
    return G6

def eq_c6(a_6):
    flp, fla, velocity_factor = hill_model()
    return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
                                        +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
                                                *velocity_factor[i][m]*a_6[m]) for m in range(len(name))])
     
x0 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_6= [(0,1) for m in muscles_list]
c_6 = ({'type': 'eq', 'fun':  eq_c6 })
result_G6=[]
for i in range(len(time)):
    result_6=optimize.minimize(G6,x0,method='SLSQP',
                              constraints=c_6,bounds=bnds_6,
                              options={'ftol': 1e-3,'maxiter': 15000})
    result_G6.append(result_6.x)  
    #print(result_6.message)
result_G6 = np.array(result_G6)                                                #/////////////   HAZI CHYBU PRI VETSI TOLERANCI
#HAZI JEDDEN BOD SPATNE pri 5kg

#-------------COMPUTING FORCE AFTER OPT. ACTIVATION IN HILL MODEL--------------
def force_6_hill():
    flp, fla, velocity_factor = hill_model()
    F6_hill = max_iso_force*(velocity_factor*fla*result_G6+flp)*np.cos(alfa)
    return F6_hill
F6_hill = force_6_hill()



#////////////////////      SD       /////////////////////

def G7(a_7):
    G7 = np.std(a_7)
    return G7


def eq_c7(a_7):
    flp, fla, velocity_factor = hill_model()
    return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
                                        +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
                                                *velocity_factor[i][m]*a_7[m]) for m in range(len(name))])
     
x0_7 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_7 = [(0,1) for m in muscles_list]
c_7= ({'type': 'eq', 'fun':  eq_c7 })
result_G7=[]
for i in range(len(time)):
    result_7=optimize.minimize(G7,x0_7,method='SLSQP',
                              constraints=c_7,bounds=bnds_7,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G7.append(result_7.x)
    x0_7=result_7.x
    print(result_7.message)
result_G7 = np.array(result_G7)


def force_7_hill():
    flp, fla, velocity_factor = hill_model()
    F7_hill = max_iso_force*(velocity_factor*fla*result_G7+flp)*np.cos(alfa)
    return F7_hill
F7_hill = force_7_hill()

# //////////       BEZ HILLA      SD////////////
def G8(force_8):  
    G8=np.std(force_8)
    return G8

def eq_c8(force_8):
    return moment_of_elbow_A[i]-np.sum(force_8*arms[i,:])
   
x0_8 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_8 = [(0, m.get_max_force()) for m in muscles_list]
c_8 = ({'type': 'eq', 'fun':  eq_c8 })
result_G8=[]
for i in range(len(time)):
    result_8=optimize.minimize(G8,x0_8,method='SLSQP',
                              constraints=c_8,bounds=bnds_8,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G8.append(result_8.x)
    x0_8=result_8.x
    # print(result_8.message)
result_G8 = np.array(result_G8)




# //////////       BEZ HILLA       NOVA FUNKCE SIGMOID ////////////
def G9(force_9):  
    G9=np.sum(1 / (1 + (1 / (2 * force_9)) ** 4))
    return G9

def eq_c9(force_9):
    return moment_of_elbow_A[i]-np.sum(force_9*arms[i,:])
   
x0_9 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_9 = [(0, m.get_max_force()) for m in muscles_list]
c_9 = ({'type': 'eq', 'fun':  eq_c9 })
result_G9=[]
for i in range(len(time)):
    result_9=optimize.minimize(G9,x0_9,method='SLSQP',
                              constraints=c_9,bounds=bnds_9,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G9.append(result_9.x)
    x0_9=result_9.x
    # print(result_8.message)
result_G9 = np.array(result_G9)


# //////////              NOVA FUNKCE SIGMOID ////////////
def G10(a_10):  
    G10=np.sum(1 / (1 + (1 / (2 * a_10)) ** 4))
    return G10

def eq_c10(a_10):
    flp, fla, velocity_factor = hill_model()
    return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
                                        +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
                                                *velocity_factor[i][m]*a_10[m]) for m in range(len(name))])
    
x0_10 = [m.get_max_force()*0.001 for m in muscles_list]
bnds_10 = [(0,1) for m in muscles_list]
c_10 = ({'type': 'eq', 'fun':  eq_c10 })
result_G10=[]
for i in range(len(time)):
    result_10=optimize.minimize(G10,x0_10,method='SLSQP',
                              constraints=c_10,bounds=bnds_10,
                              options={'ftol': 1e-7,'maxiter': 1500})
    result_G10.append(result_10.x)
    x0_10=result_10.x
    # print(result_9.message)
result_G10 = np.array(result_G10)

def force_10_hill():
    flp, fla, velocity_factor = hill_model()
    F10_hill = max_iso_force*(velocity_factor*fla*result_G10+flp)*np.cos(alfa)
    return F10_hill
F10_hill = force_10_hill()



# # //////////              NOVA FUNKCE SIGMOID ////////////
# def G9(a_9):  
#     G9=np.sum(1 / (1 + (1 / (2 * a_9)) ** 2))
#     return G9

# def eq_c9(a_9):
#     flp, fla, velocity_factor = hill_model()
#     return -moment_of_elbow_A[i]+np.sum([np.sum(max_iso_force[m]*flp[i][m]*np.cos(alfa[m])*arms[i][m])
#                                         +np.sum(max_iso_force[m]*fla[i][m]*np.cos(alfa[m])*arms[i][m]
#                                                 *velocity_factor[i][m]*a_9[m]) for m in range(len(name))])
    
# x0_9 = [0.001,0.001,0.001,0.001,0.001,0.001,0.001]
# bnds = [(0,1) for m in muscles_list]
# c_9 = ({'type': 'eq', 'fun':  eq_c9 })
# result_G9=[]
# for i in range(len(time)):
#     result_9=optimize.minimize(G9,x0_9,method='SLSQP',
#                               constraints=c_9,bounds=bnds,
#                               options={'ftol': 1e-4,'maxiter': 1500})
#     result_G9.append(result_9.x)
#     x0_9=result_9.x
#     #print(result_6.message)
# result_G9 = np.array(result_G9)

# def force_9_hill():
#     flp, fla, velocity_factor = hill_model()
#     F9_hill = max_iso_force*(velocity_factor*fla*result_G9+flp)*np.cos(alfa)
#     return F9_hill
# F9_hill = force_9_hill()





