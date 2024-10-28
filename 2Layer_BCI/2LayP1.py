import numpy as np
import dedalus.public as d3
from dedalus.tools.parallel import Sync
import h5py

import time
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
Ro = 0.1
Bu = 8

Ly_instab = 8
Ly = Ly_instab*4
Ny = 256
Lx = Ly*2
Nx = Ny*2

dealias = 3/2
stop_sim_time = 300
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
delx = Lx/Nx
# nu8 = 1*delx**8
nu4 = (delx)**4*3

if_ridig = 2

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

# Fields
#################
q1 = dist.Field(name='Q1_init', bases=(xbasis,ybasis) )
q2 = dist.Field(name='Q2_init', bases=(xbasis,ybasis) )

Q1M = dist.Field(name='Q1_mean', bases=ybasis )
Q2M = dist.Field(name='Q2_mean', bases=ybasis )

#################
P0_1 = dist.Field(name='P0_1', bases=(xbasis,ybasis) )
P0_2 = dist.Field(name='P0_2', bases=(xbasis,ybasis) )
tau_P0_1 = dist.Field(name='tau_P0_1')
tau_P0_2 = dist.Field(name='tau_P0_2')

#################
P1_1 = dist.Field(name='P1_1', bases=(xbasis,ybasis) )
P1_2 = dist.Field(name='P1_2', bases=(xbasis,ybasis) )
tau_P1_1 = dist.Field(name='tau_P1_1')
tau_P1_2 = dist.Field(name='tau_P1_2')

#################
F1_1 = dist.Field(name='F1_1', bases=(xbasis,ybasis) )
F1_2 = dist.Field(name='F1_2', bases=(xbasis,ybasis) )
tau_F1_1 = dist.Field(name='tau_F1_1')
tau_F1_2 = dist.Field(name='tau_F1_2')

#################
G1_1 = dist.Field(name='G1_1', bases=(xbasis,ybasis) )
G1_2 = dist.Field(name='G1_2', bases=(xbasis,ybasis) )
# tau_G1 = dist.Field(name='tau_G1')
tau_G1_1 = dist.Field(name='tau_G1_1')
tau_G1_2 = dist.Field(name='tau_G1_2')

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
lap = lambda A: d3.Laplacian(A)
xinteg = lambda A: d3.Integrate(A, ('x'))
integ = lambda A: d3.Integrate(A, ('x', 'y'))
xavg = lambda A: d3.Average(A, ('x'))
avg = lambda A: d3.Average(A, ('x', 'y'))

lp4 = lambda A: lap(lap(A))

x, y = dist.local_grids(xbasis, ybasis)

J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)

###
y_mat = dist.Field(name='y_mat', bases=(xbasis,ybasis) )
y_mat['g'] = y

q1_nomean = q1-avg(q1)
q2_nomean = q2-avg(q2)

u1 = -dy(P0_1)+Ro*(-dy(P1_1)-F1_1)
v1 =  dx(P0_1)+Ro*( dx(P1_1)-G1_1)
u2 = -dy(P0_2)+Ro*(-dy(P1_2)-F1_2)
v2 =  dx(P0_2)+Ro*( dx(P1_2)-G1_2)
h1 = if_ridig*P0_1-P0_2+Ro*(if_ridig*P1_1-P1_2-dx(G1_1)+dy(F1_1))
h2 = P0_2-P0_1+Ro*(P1_2-P1_1-dx(G1_2)+dy(F1_2))

zeta_1 = -dy(u1)+dx(v1)
zeta_2 = -dy(u2)+dx(v2)
div_1 = dx(u1)+dy(v1)
div_2 = dx(u2)+dy(v2)

KE1 = avg(u1**2+v1**2)*0.5
KE2 = avg(u2**2+v2**2)*0.5
zeta1_skew = avg(zeta_1**3)/( avg(zeta_1**2)**(3/2)+1e-13 )
zeta2_skew = avg(zeta_2**3)/( avg(zeta_2**2)**(3/2)+1e-13 )

# Problem
problem = d3.IVP([q1, q2, \
                  P0_1, P0_2, tau_P0_1, tau_P0_2, \
                  P1_1, P1_2, tau_P1_1, tau_P1_2, \
                  F1_1, F1_2, tau_F1_1, tau_F1_2, \
                  G1_1, G1_2, tau_G1_1, tau_G1_2, \
                    ], namespace=locals())

#################
problem.add_equation("dt(q1) + nu4*lp4(q1) = - (u1*dx(q1)+v1*dy(q1))")
problem.add_equation("dt(q2) + nu4*lp4(q2) = - (u2*dx(q2)+v2*dy(q2))")

#################
problem.add_equation("lap(P0_1)+(P0_2-if_ridig*P0_1)/Bu +tau_P0_1 = q1_nomean")
problem.add_equation("lap(P0_2)+(P0_1-P0_2)/Bu          +tau_P0_2 = q2_nomean")
problem.add_equation("integ(P0_1)=0")
problem.add_equation("integ(P0_2)=0")

#################
problem.add_equation("lap(P1_1)+(P1_2-if_ridig*P1_1)/Bu +tau_P1_1 = -(if_ridig*P0_1-P0_2)**2/(Bu**2)+lap(P0_1)*(if_ridig*P0_1-P0_2)/Bu")
problem.add_equation("lap(P1_2)+(P1_1-P1_2)/Bu          +tau_P1_2 = -(P0_2-P0_1)**2/(Bu**2)         +lap(P0_2)*(P0_2-P0_1)/Bu")
problem.add_equation("integ(P1_1)=0")
problem.add_equation("integ(P1_2)=0")

#################
problem.add_equation("lap(F1_1)+(F1_2-if_ridig*F1_1)/Bu +tau_F1_1 = ( J(dx(P0_1+P0_2),P0_1-P0_2)+(if_ridig-1)*J(dx(P0_1),P0_1) )/Bu")
problem.add_equation("lap(F1_2)+(F1_1-F1_2)/Bu          +tau_F1_2 = ( J(dx(P0_1+P0_2),P0_2-P0_1)                               )/Bu")
problem.add_equation("integ(F1_1)=0")
problem.add_equation("integ(F1_2)=0")

#################
problem.add_equation("lap(G1_1)+(G1_2-if_ridig*G1_1)/Bu +tau_G1_1 = ( J(dy(P0_1+P0_2),P0_1-P0_2)+(if_ridig-1)*J(dy(P0_1),P0_1) )/Bu")
problem.add_equation("lap(G1_2)+(G1_1-G1_2)/Bu          +tau_G1_2 = ( J(dy(P0_1+P0_2),P0_2-P0_1)                               )/Bu")
problem.add_equation("integ(G1_1)=0")
problem.add_equation("integ(G1_2)=0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
ICname = 'MeanPV_%.2f_%.3f_%d/MeanPV_%.2f_%.3f_%d_s%i' %(Ly_instab,Ro,Nx, Ly_instab,Ro,Nx, 1)
ICname = ICname.replace(".", "d" ); ICname = ""+ICname+'.h5'
write = solver.load_state(ICname, index=1, allow_missing=True)

q1.change_scales(1); q2.change_scales(1)
Q1M['g'] = q1['g'][0,:]
Q2M['g'] = q2['g'][0,:]

q1['g'] = 0; q2['g'] = 0
q1.fill_random('c', seed=42, distribution='normal', scale=1e-3) # Random noise
q1.low_pass_filter(shape=(20, 20)); q1.high_pass_filter(shape=(5, 5))
q2.fill_random('c', seed=69, distribution='normal', scale=1e-3) # Random noise
q2.low_pass_filter(shape=(20, 20)); q2.high_pass_filter(shape=(5, 5))

q1['g'] += Q1M['g']; q2['g'] += Q2M['g']
# q1['g'] = q1_nomean.evaluate()['g']; q2['g'] = q2_nomean.evaluate()['g']; 

# Analysis
snapname = '2LayP1_sp_%.2f_%.3f_%d' %(Ly_instab,Ro,Nx)
snapname = snapname.replace(".", "d" ); 
snapdata = solver.evaluator.add_file_handler(snapname, sim_dt=1, max_writes=10)
snapdata.add_task(-(-q1), name='q1')
snapdata.add_task(-(-q2), name='q2')
snapdata.add_task(-(-P0_1), name='P0_1')
snapdata.add_task(-(-P0_2), name='P0_2')
snapdata.add_task(-(-P1_1), name='P1_1')
snapdata.add_task(-(-P1_2), name='P1_2')
# snapdata.add_task(-(-P0_1_c), name='P0_1_c')
# snapdata.add_task(-(-P0_2_c), name='P0_2_c')
# snapdata.add_task(-(-P1_1_c), name='P1_1_c')
# snapdata.add_task(-(-P1_2_c), name='P1_2_c')
snapdata.add_task(-(-h1), name='h1')
snapdata.add_task(-(-h2), name='h2')
snapdata.add_task(-(-u1), name='u1')
snapdata.add_task(-(-u2), name='u2')
snapdata.add_task(-(-zeta_1), name='zeta_1')
snapdata.add_task(-(-zeta_2), name='zeta_2')
snapdata.add_task(-(-div_1), name='div_1')
snapdata.add_task(-(-div_2), name='div_2')


diagname = '2LayP1_dg_%.2f_%.3f_%d' %(Ly_instab,Ro,Nx)
diagname = diagname.replace(".", "d" ); 
diagdata = solver.evaluator.add_file_handler(diagname, sim_dt=0.1, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')
diagdata.add_task(avg(q1), name='q1_avg')
diagdata.add_task(avg(q2), name='q2_avg')
diagdata.add_task(zeta1_skew, name='zeta1_skew')
diagdata.add_task(zeta2_skew, name='zeta2_skew')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(u1), name='absu1')
flow_cfl.add_property(abs(v1), name='absv1')
flow_cfl.add_property(abs(u2), name='absu2')
flow_cfl.add_property(abs(v2), name='absv2')

print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property( (u1**2+v1**2)*0.5 , name='KE1')

# Main loop
timestep = 1e-7; 

delx = Lx/Nx; dely = Ly/Ny
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)

        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max(1e-10,flow_cfl.max('absu1'),flow_cfl.max('absu2')); maxV = max(1e-10,flow_cfl.max('absv1'),flow_cfl.max('absv2'))
            timestep_CFL = min(delx/maxU,dely/maxV)*0.5
            timestep = min(max(1e-10, timestep_CFL), 0.1)
            
        if (solver.iteration-1) % print_freq == 0:
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE1=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE1')))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()