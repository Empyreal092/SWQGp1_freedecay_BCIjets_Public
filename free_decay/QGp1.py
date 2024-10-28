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
q  = dist.Field(name='q' , bases=(xbasis,ybasis))
P0 = dist.Field(name='P0', bases=(xbasis,ybasis))
P1 = dist.Field(name='P1', bases=(xbasis,ybasis))
F1 = dist.Field(name='F1', bases=(xbasis,ybasis))
G1 = dist.Field(name='G1', bases=(xbasis,ybasis))

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

x, y = dist.local_grids(xbasis, ybasis)

avg =  lambda A: d3.Average(A, ('x','y'))
lap = lambda A: d3.Laplacian(A)
spo = lambda A: lap(A)-A
lp8 = lambda A: lap(lap(A))
J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)

u = -dy(P0)+Ro*(-dy(P1)-F1)
v =  dx(P0)+Ro*( dx(P1)-G1)
h_raw = P0+Ro*(P1-dx(G1)+dy(F1))
h = h_raw-avg(h_raw)

q_nomean = q-avg(q)

zeta = -dy(u)+dx(v)
div = dx(u)+dy(v)

# Problem
problem = d3.IVP([q, P0, P1, F1, G1], namespace=locals())

problem.add_equation("spo(P0) = q_nomean")
problem.add_equation("spo(P1) = -P0*P0+lap(P0)*P0")
problem.add_equation("spo(F1) = J(dx(P0),P0)")
problem.add_equation("spo(G1) = J(dy(P0),P0)")

problem.add_equation("dt(q) + nu*lp8(q) = - ( u*dx(q)+v*dy(q) )")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
ICname = 'QGp1_IC_%.2f_%d/QGp1_IC_%.2f_%d_s%i' %(Ro,rand_seed, Ro,rand_seed, 1)
ICname = ICname.replace(".", "d" ); ICname = ""+ICname+'.h5'
write = solver.load_state(ICname, index=2, allow_missing=True)
KE = d3.Average( (u**2+v**2)*(1+Ro*h)/2, ('x', 'y'))
PE = d3.Average( h**2/2, ('x', 'y'))
KE0 = d3.Average( -P0*lap(P0)/2, ('x', 'y'))
PE0 = d3.Average( P0**2/2, ('x', 'y'))
Ens = d3.Average( (1+Ro*h)*q**2  /2, ('x', 'y'))
zeta_skew = avg(zeta**3)/( avg(zeta**2)**(3/2) )
q_skew = avg(q_nomean**3)/( avg(q_nomean**2)**(3/2) )

# Analysis
data_name = 'QGp1_sp_%.2f_%d' %(Ro,rand_seed)
data_name = data_name.replace(".", "d" )
snapshots = solver.evaluator.add_file_handler(data_name, sim_dt=1, max_writes=20)
snapshots.add_task(-(-u), name='u')
snapshots.add_task(-(-v), name='v')
snapshots.add_task(zeta, name='zeta')
snapshots.add_task(div, name='div')
snapshots.add_task(-(-q), name='PV')
snapshots.add_task(-(-h), name='h')

# Analysis
data_name = 'QGp1_dg_%.2f_%d' %(Ro,rand_seed)
data_name = data_name.replace(".", "d" )
diagsave = solver.evaluator.add_file_handler(data_name, sim_dt=0.1, max_writes=3e5)
diagsave.add_task(KE, name='KE')
diagsave.add_task(PE, name='PE')
diagsave.add_task(KE0, name='KE0')
diagsave.add_task(PE0, name='PE0')
diagsave.add_task(Ens, name='Ens')
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
            maxU = max(1e-10,max(flow_cfl.max('absu'),flow_cfl.max('absv')))
            timestep_CFL = delx/maxU*0.5
            timestep = min(max(1e-7, timestep_CFL), 1)
        if (solver.iteration-1) % 50 == 0:
            logger.info('Iteration=%i, Time=%f, dt=%.3e, KE=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE')))
    
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()