import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
log_n = 10
k_peak = 6
Lx, Ly = k_peak*2*np.pi, k_peak*2*np.pi
# Lx, Ly = 2*np.pi, 2*np.pi
Nx, Ny = 2**log_n, 2**log_n
delx = Lx/Nx

dealias = 3/2
stop_sim_time = 500
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
Ro = float(sys.argv[1])/100
# Ro = 0.05
print('Ro=%f' %Ro)

rand_seed = int(sys.argv[2])
print('rand_seed=%d' %rand_seed)

nu = (delx/1.0)**4*3

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

# Fields
u  = dist.Field(name='u', bases=(xbasis,ybasis))
v  = dist.Field(name='v', bases=(xbasis,ybasis))
h  = dist.Field(name='h', bases=(xbasis,ybasis))

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

x, y = dist.local_grids(xbasis, ybasis)

avg =  lambda A: d3.Average(A, ('x','y'))
lap = lambda A: d3.Laplacian(A)
lp8 = lambda A: lap(lap(A))

J   = lambda A,B: dx(A)*dy(B)-dy(A)*dx(B)

zeta = -dy(u)+dx(v)
div  =  dx(u)+dy(v)
q = ((1+Ro*zeta)/(1+Ro*h)-1)/Ro
q_nomean = q-avg(q)

# Problem
problem = d3.IVP([u, v, h], namespace=locals())
problem.add_equation("dt(u) + nu*lp8(u) + 1/Ro*(dx(h)-v) = - (u*dx(u)+v*dy(u))")
problem.add_equation("dt(v) + nu*lp8(v) + 1/Ro*(dy(h)+u) = - (u*dx(v)+v*dy(v))")
problem.add_equation("dt(h) + nu*lp8(h) + 1/Ro*(dx(u)+dy(v)) = - (dx(h*u)+dy(h*v))")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
ICname = 'SW_IC_%.2f_%d/SW_IC_%.2f_%d_s%i' %(Ro,rand_seed, Ro,rand_seed, 1)
ICname = ICname.replace(".", "d" ); ICname = ""+ICname+'.h5'
write = solver.load_state(ICname, index=2, allow_missing=True)

KE_sub = d3.Average( (u**2+v**2)*(1+Ro*h)/2, ('x', 'y'))
PE_sub = d3.Average( h**2/2, ('x', 'y'))
Ens_sub = d3.Average( (1+Ro*h)*q**2  /2, ('x', 'y'))
zeta_skew = avg(zeta**3)/( avg(zeta**2)**(3/2) )
q_skew = avg(q_nomean**3)/( avg(q_nomean**2)**(3/2) )

# Analysis
data_name = 'SW_sp_%.2f_%d' %(Ro,rand_seed)
data_name = data_name.replace(".", "d" )
snapshots = solver.evaluator.add_file_handler(data_name, sim_dt=1, max_writes=20)
snapshots.add_task(-(-u), name='u')
snapshots.add_task(-(-v), name='v')
snapshots.add_task(zeta, name='zeta')
snapshots.add_task(div, name='div')
snapshots.add_task(-(-h), name='h')
snapshots.add_task(-(-q), name='q')

data_name = 'SW_dg_%.2f_%d' %(Ro,rand_seed)
data_name = data_name.replace(".", "d" )
diagsave = solver.evaluator.add_file_handler(data_name, sim_dt=0.1, max_writes=3e5)
diagsave.add_task(-(-KE_sub), name='KE')
diagsave.add_task(-(-PE_sub), name='PE')
diagsave.add_task(-(-Ens_sub), name='Ens')
diagsave.add_task(zeta_skew, name='zeta_skew')
diagsave.add_task(q_skew, name='q_skew')
diagsave.add_task(avg(q), name='q_mean')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(u), name='absu')
flow_cfl.add_property(abs(v), name='absv')

print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow.add_property(u**2+v**2, name='KE')

# Main loop
timestep = 1e-7
delx = Lx/Nx

try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max( 1e-10, max(flow_cfl.max('absu'),flow_cfl.max('absv')) ); 
            timestep_CFL = delx/maxU*0.5; 
            timestep = min(max(1e-7, timestep_CFL), 1)
        if (solver.iteration-1) % print_freq == 0:
            logger.info('Iteration=%i, Time=%f, dt=%.3e, KE=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE')))
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

