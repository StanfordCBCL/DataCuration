import pdb
import numpy as np
from network_util_NR import *
import matplotlib.pyplot as plt

def run_network_util(block_list, deltat, time, bc_type):
    connect_list, wdict = connect_blocks_by_inblock_list(block_list)
    neq = compute_neq(block_list, wdict)  # compute number of equations
    for b in block_list:  # run a consistency check
        check_block_connection(b)
    var_name_list = assign_global_ids(block_list, wdict)  # assign solution variables with global ID

    y0, ydot0 = initialize_solution_structures(neq)  # initialize solution structures
    curr_y = y0.copy()
    curr_ydot = ydot0.copy()

    rho = 0.1
    args = {}
    args['Time step'] = deltat
    args['rho'] = rho
    args['Wire dictionary'] = wdict

    ylist = [curr_y.copy()]
    for t in time[1:]:
        args['Solution'] = curr_y
        curr_y, curr_ydot = gen_alpha_dae_integrator_NR(curr_y, curr_ydot, t, block_list, args, deltat, rho)
        ylist.append(curr_y)

    ###############
    # these corrections are needed based on how i think the above for loop works: the "t" in the above for loop corresponds to t_current (t_n) and the output of gen_alpha_dae_integrator_NR is y_next (y_(n+1)). see https://github.com/StanfordCBCL/0D_LPN_Python_Solver/blob/master/test_nonlin_res.py for proof.
    ylist = ylist[:-1]
    ylist.insert(0, y0.copy())
    ###############

    ylist = np.array(ylist)
    for i in range(len(ylist[0, :])):
        if var_name_list[i] == "P_inflow_" + bc_type:
            inlet_pressure = ylist[:, i]
            return inlet_pressure

def run_rcr(Qfunc, time, p, distal_pressure):
    deltat = time[1] - time[0]

    inflow = UnsteadyFlowRef(connecting_block_list=['rcr'], Qfunc=Qfunc, name='inflow', flow_directions=[+1])
    rcr = RCRBlock(connecting_block_list=['inflow', 'ground'], Rp=p['Rp'], C=p['C'], Rd=p['Rd'], name='rcr',
                   flow_directions=[-1, +1])
    ground = PressureRef(connecting_block_list=['rcr'], Pref=distal_pressure, name='ground', flow_directions=[-1])
    block_list = [inflow, rcr, ground]

    return run_network_util(block_list, deltat, time, bc_type = "rcr")


def run_coronary(Qfunc, time, p, p_v_time, p_v_pres, cardiac_cycle_period):
    # compute intramyocadial pressure derivative
    p_v_time = p_v_time.tolist()
    p_v_pres = p_v_pres.tolist()
    dPvdt_f = np.zeros((len(p_v_time), 2))
    dPvdt_f[:, 0] = p_v_time
    deltat_temp = p_v_time[1] - p_v_time[0]
    extended_time = np.array([p_v_time[0] - deltat_temp] + p_v_time + [p_v_time[-1] + deltat_temp]) # extend time beyond the first and last boundaries to get a periodic dPvdt
    p_v_pres_extended = np.array([p_v_pres[-2]] + p_v_pres + [p_v_pres[1]])
    dPvdt_f[:, 1] = np.gradient(p_v_pres_extended, extended_time)[1:-1]

    distal_pressure = 0.0
    deltat = time[1] - time[0]

    inflow = UnsteadyFlowRef(connecting_block_list=['coronary'], Qfunc=Qfunc, name='inflow', flow_directions=[+1])
    coronary = OpenLoopCoronaryWithDistalPressureBlock(connecting_block_list=['inflow'], R1=p['R1'], C1=p['C1'],
                                                       R2=p['R2'], C2=p['C2'], R3=p['R3'], dPvdt_f=dPvdt_f,
                                                       cardiac_cycle_period=cardiac_cycle_period, Pv=distal_pressure,
                                                       name='coronary', flow_directions=[-1])
    block_list = [inflow, coronary]

    return run_network_util(block_list, deltat, time, bc_type = "coronary")
