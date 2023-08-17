import numpy as np
from muscle import Muscle
from scipy import optimize
from inputs import *
from base_computing_system import hill_model

from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

arms_list = []
muscles_list = []
for i in range(len(name)):
    muscle = Muscle(proximal_attachment[i], distal_attachment[i], alfa[i],
                    optimal_fiber_length[i], max_iso_force[i], name[i],
                    tendon_length[i], density_MS[i], weight[i])
    muscles_list.append(muscle)
    arms_list.append(muscle.arm())
    arms = (np.transpose(np.squeeze(arms_list)))
arms[:, -3:] = -0.02
# /////////////////////          GTO MODELING             //////////////////////

hustota = np.transpose(
    np.array([muscle.density_MS for muscle in muscles_list]))
mi = np.transpose(np.array([muscle.weight for muscle in muscles_list]))

# /////////////////////     LINEAR MODEL         ///////////////////////////////

# ---------    OPTIMATIZATION OF FORCES OF MUSCLES through density   -----------


# #---------    OPTIMATIZATION OF activation through density   HILL/-----------


def fun_rho(rho):
    #rho = np.append(rho, 100-np.sum(rho))
    print(rho)

    def G6(a_6):
        G6 = np.sum(1/(1+np.power((1/2/a_6), 4)) * rho)
     
        return G6

    def eq_c6(a_6):
        flp, fla, velocity_factor = hill_model()
        return -moment_of_elbow_A[i]+np.sum(max_iso_force*(velocity_factor[i, :]*fla[i, :]*a_6+flp[i, :])*np.cos(alfa)*arms[i][:])

    x0 = [m.get_max_force()*0.001 for m in muscles_list]
    bnds = [(0, 1) for m in muscles_list]
    c_6 = ({'type': 'eq', 'fun':  eq_c6})
    result = 0
    for i in range(len(time)):
        result_6 = optimize.minimize(G6, x0, method='SLSQP',
                                     constraints=c_6, bounds=bnds,
                                     options={'ftol': 1e-3, 'maxiter': 1500})
        print(result_6.message)
        x0 = result_6.x
        xfun = result_6.fun
        result = result + xfun

    return result


def fun_rho_res(rho):

    #rho = np.append(rho, 100-np.sum(rho))
    print(rho)

    def G6(a_6):
        G6 = np.sum(1/(1+np.power((1/2/a_6), 4)) * rho)
       
        return G6

    def eq_c6(a_6):
        flp, fla, velocity_factor = hill_model()
        return -moment_of_elbow_A[i]+np.sum(max_iso_force*(velocity_factor[i, :]*fla[i, :]*a_6+flp[i, :])*np.cos(alfa)*arms[i][:])

    x0 = [m.get_max_force()*0.5 for m in muscles_list]
    bnds = [(0, 1) for m in muscles_list]
    c_6 = ({'type': 'eq', 'fun':  eq_c6})
    result = 0
    result_G2 = []
    for i in range(len(time)):
        result_6 = optimize.minimize(G6, x0, method='SLSQP',
                                     constraints=c_6, bounds=bnds,
                                     options={'ftol': 1e-3, 'maxiter': 1500})
        print(result_6.message)
        x0 = result_6.x
        xfun = result_6.fun
        result = result + xfun
        result_G2.append(result_6.x)

    return result, result_G2, rho


def fun_eq(rho):
    return np.sum(rho) - 1
    


#x0 = [1/7 for m in muscles_list]
x0 = [0.25, 0.25, 0.25, 0.25, 0, 0, 0]
bnds = [(0.1, 1) for m in x0]
c_rho = ({'type': 'eq', 'fun':  fun_eq})
result_rho = optimize.minimize(fun_rho, x0, method='SLSQP', bounds=bnds, constraints=c_rho,
                               options={'ftol': 1e-3, 'maxiter': 1500})


result, result_G, rho = fun_rho_res(result_rho.x)


def force_hill():
    flp, fla, velocity_factor = hill_model()
    F2_hill = max_iso_force*(velocity_factor*fla*result_G+flp)*np.cos(alfa)
    return F2_hill


F_hill = force_hill()


colors = ['#FD3216', '#511CFB', '#1CA71C', '#FE00FA', '#B68100', '#FECB52', '#19D3F3']

# #                          "G11-
# fig = go.Figure()
# for i in range(len(name)):
#     y_data = np.vstack(result_G)[:, i]
#     #[3:]

#     fig.add_trace(go.Scatter(x=time[3:], y=y_data,
#                              mode='lines', name=name[i], line=dict(width=3, color=colors[i]),
#                              marker=dict(size=6),
#                              #showlegend=False
#                              ))
    
# fig.update_layout(title='',
#                   xaxis_title='Time [s]',
#                   yaxis_title='a [-]',
#                   template='plotly_white')
# fig.update_xaxes(showline=True, linewidth=3, linecolor='black')
# fig.update_yaxes(showline=True, linewidth=3, linecolor='black')
# fig.update_layout(width=950, height=550, autosize=True)
# fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99))
# fig.update_layout(font=dict(family="Century", color="black", size=30))
# fig.update_layout(legend_font=dict(size=30))
# fig.show()
# pio.write_html(fig, file="./bez/G112.html", auto_open=True)
# fig.write_image("./bez/G112.jpeg", format='jpeg')

fig = go.Figure()
for i in range(len(name)):
    
    
    y_data = F_hill[:, i]
    #[3:]

    fig.add_trace(go.Scatter(x=time
                             #[3:]
                             , y=y_data,
                             mode='lines', name=name[i], line=dict(width=3, color=colors[i]),
                             marker=dict(size=6),
                             #showlegend=False
                             ))
fig.update_layout(title='',
                  xaxis_title='Time [s]',
                  yaxis_title='F [N]',
                  template='plotly_white')
fig.update_xaxes(showline=True, linewidth=3, linecolor='black')
fig.update_yaxes(showline=True, linewidth=3, linecolor='black')
fig.update_layout(width=950, height=550, autosize=True)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99))
fig.update_layout(font=dict(family="Century", color="black", size=30))
fig.update_layout(legend_font=dict(size=30))
fig.show()
pio.write_html(fig, file="./bez/G112_F_k4.html", auto_open=True)
fig.write_image("./bez/G112_F_k4.jpeg", format='jpeg')
