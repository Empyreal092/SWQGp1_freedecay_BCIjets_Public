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
u1 = dist.Field(name='u1', bases=(xbasis,ybasis) )
u2 = dist.Field(name='u2', bases=(xbasis,ybasis) )

#################
v1 = dist.Field(name='v1', bases=(xbasis,ybasis) )
v2 = dist.Field(name='v2', bases=(xbasis,ybasis) )

#################
h1 = dist.Field(name='h1', bases=(xbasis,ybasis) )
h2 = dist.Field(name='h2', bases=(xbasis,ybasis) )
h1M = dist.Field(name='h1M', bases=ybasis )
h2M = dist.Field(name='h2M', bases=ybasis )

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
zeta_1 = -dy(u1)+dx(v1)
zeta_2 = -dy(u2)+dx(v2)
div_1 = dx(u1)+dy(v1)
div_2 = dx(u2)+dy(v2)

KE1 = avg(u1**2+v1**2)*0.5
KE2 = avg(u2**2+v2**2)*0.5
# PE1 = avg(h1**2)*0.5
# PE2 = avg(h2**2)*0.5
zeta1_skew = avg(zeta_1**3)/( avg(zeta_1**2)**(3/2) )
zeta2_skew = avg(zeta_2**3)/( avg(zeta_2**2)**(3/2) )

Rc = 1/Ro

# Problem
problem = d3.IVP([u1, u2, \
                  v1, v2, \
                  h1, h2
                    ], namespace=locals())

#################
problem.add_equation("dt(u1) + nu4*lp4(u1) + Rc*(-v1+dx(h1+  h2)) = - (u1*dx(u1)+v1*dy(u1)) ")
problem.add_equation("dt(u2) + nu4*lp4(u2) + Rc*(-v2+dx(h1+if_ridig*h2)) = - (u2*dx(u2)+v2*dy(u2)) ")
# problem.add_equation("dt(u2) + Rc*(-v2+dx(h1+  h2)) + nu2*lap(u2) +lift(tau_u2_b,-1)+lift(tau_u2_t,-2) = -res_cons*(xavg(u2)-U2M) - (u2*dx(u2)+v2*dy(u2)) - (0)")

#################
problem.add_equation("dt(v1) + nu4*lp4(v1) + Rc*(u1+dy(h1+  h2)) = - (u1*dx(v1)+v1*dy(v1)) ")
problem.add_equation("dt(v2) + nu4*lp4(v2) + Rc*(u2+dy(h1+if_ridig*h2)) = - (u2*dx(v2)+v2*dy(v2)) ")
# problem.add_equation("dt(v2) + Rc*(u2+dy(h1+  h2)) + nu2*lap(v2) +lift(tau_v2_b,-1)+lift(tau_v2_t,-2) = -res_cons*(xavg(v2)-0) - (u2*dx(v2)+v2*dy(v2)) - (0)")

#################
problem.add_equation("dt(h1) + nu4*lp4(h1) + Bu*Rc*(dx(u1)+dy(v1)) = - ( dx(u1*h1)+dy(v1*h1) ) ")
problem.add_equation("dt(h2) + nu4*lp4(h2) + Bu*Rc*(dx(u2)+dy(v2)) = - ( dx(u2*h2)+dy(v2*h2) ) ")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
ICname = 'MeanSW_%.2f_%.3f_%d/MeanSW_%.2f_%.3f_%d_s%i' %(Ly_instab,Ro,Nx, Ly_instab,Ro,Nx, 1)
ICname = ICname.replace(".", "d" ); ICname = ""+ICname+'.h5'
write = solver.load_state(ICname, index=2, allow_missing=True)

h1.change_scales(1); h2.change_scales(1)
h1M['g'] = h1['g'][0,:]; h2M['g'] = h2['g'][0,:]

h1.fill_random('c', seed=42, distribution='normal', scale=1e-3) # Random noise
h1.low_pass_filter(shape=(20, 20)); h1.high_pass_filter(shape=(5, 5))
h2.fill_random('c', seed=69, distribution='normal', scale=1e-3) # Random noise
h2.low_pass_filter(shape=(20, 20)); h2.high_pass_filter(shape=(5, 5))
h1['g'] += h1M['g']; h2['g'] += h2M['g']

v1['g'] = 0; v2['g'] = 0

# Analysis
snapname = '2LaySW_sp_%.2f_%.3f_%d' %(Ly_instab,Ro,Nx)
snapname = snapname.replace(".", "d" ); 
snapdata = solver.evaluator.add_file_handler(snapname, sim_dt=1, max_writes=50)
# snapdata.add_task(-(-u1), name='u1')
# snapdata.add_task(-(-u2), name='u2')
# snapdata.add_task(-(-v1), name='v1')
# snapdata.add_task(-(-v2), name='v2')
snapdata.add_task(-(-h1), name='h1')
snapdata.add_task(-(-h2), name='h2')
snapdata.add_task(-(-zeta_1), name='zeta_1')
snapdata.add_task(-(-zeta_2), name='zeta_2')
snapdata.add_task(-(-div_1), name='div_1')
snapdata.add_task(-(-div_2), name='div_2')

diagname = '2LaySW_dg_%.2f_%.3f_%d' %(Ly_instab,Ro,Nx)
diagname = diagname.replace(".", "d" ); 
diagdata = solver.evaluator.add_file_handler(diagname, sim_dt=0.1, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')
diagdata.add_task(zeta1_skew, name='zeta1_skew')
diagdata.add_task(zeta2_skew, name='zeta2_skew')

# CFL
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property( abs(u1), name='absu1')
flow_cfl.add_property( abs(v1), name='absv1')
flow_cfl.add_property( abs(u2), name='absu2')
flow_cfl.add_property( abs(v2), name='absv2')

# Flow properties
print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property( (u1**2+v1**2)/2 , name='KE__1')

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
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE1=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE__1')))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()