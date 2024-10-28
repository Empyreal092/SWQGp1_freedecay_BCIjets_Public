import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, '../../subroutines')
from isospectrum import isospectrum

for rand_seed in range(1,11):

    #Physical Parameters
    Ro = float(sys.argv[1])/100
    print('Ro=%f' %Ro)

    # Numerics Parameters
    log_n = 8
    k_peak = 6
    Lx, Ly = k_peak*2*np.pi, k_peak*2*np.pi
    Nx, Ny = 2**log_n, 2**log_n

    dealias = 3/2
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

    # Fields
    psi  = dist.Field(bases=(xbasis,ybasis))

    h  = dist.Field(bases=(xbasis,ybasis))
    tau_h = dist.Field()

    # Substitutions
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])

    x, y = dist.local_grids(xbasis, ybasis)

    lap = lambda A: d3.Laplacian(A)

    J   = lambda A,B: dx(A)*dy(B)-dy(A)*dx(B)

    # Inisital psi ###########
    psi.fill_random('c', seed=rand_seed, distribution='normal', scale=10) # Random noise
    # psi.fill_random('c', distribution='normal', scale=1e6) # Random noise
    psi.low_pass_filter(shape=(k_peak*2*1.6+3, k_peak*2*1.6+3));  psi.high_pass_filter(shape=(k_peak*2*1.6, k_peak*2*1.6))

    # kx = xbasis.wavenumbers; ky = ybasis.wavenumbers; 
    # kx2, ky2 = np.meshgrid(kx, ky)
    # KK2 = np.sqrt(kx2**2+ky2**2)

    # m = 25; 
    # # k0 = k_peak
    # k0 = 1
    # psi['c'] *= (KK2**(m/2)/(KK2+k0)**m)**(1/2)
    # psi['c'][KK2!= 0] /= KK2[KK2!= 0]

    KK = xbasis.wavenumbers[1::2]
    zeta_mag2d = np.squeeze(psi['c'])*np.conj(np.squeeze(psi['c']))
    zeta_spec = isospectrum(zeta_mag2d); 

    KE_mean = np.sum(zeta_spec*KK**2)

    psi['c'] *= np.sqrt(1/KE_mean)*1

    # Solve for h ###########
    # Problem
    problem = d3.LBVP([h, tau_h], namespace=locals())
    problem.add_equation("lap(h)+tau_h = lap(psi) + 2*Ro*J(dx(psi),dy(psi))")
    problem.add_equation("integ(h) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve()

    # Solve for psi_t ###########
    chi_old = dist.Field(bases=(xbasis,ybasis))
    chi  = dist.Field(bases=(xbasis,ybasis))
    tau_chi = dist.Field()

    psi_t = dist.Field(bases=(xbasis,ybasis))
    tau_psi_t = dist.Field()

    chi['g'] = 0

    chi_old.change_scales(1); chi.change_scales(1)
    chi_old['g'] = chi['g']

    prob_psit = d3.LBVP([psi_t, tau_psi_t], namespace=locals())
    prob_psit.add_equation("lap(psi_t)+tau_psi_t = -1*(J(psi,lap(psi)) + Ro**(-1)*lap(chi_old) + div(lap(psi)*grad(chi_old)))")
    prob_psit.add_equation("integ(psi_t) = 0")

    # Solver
    solv_psit = prob_psit.build_solver()
    solv_psit.solve()

    # Solve for chi ###########
    J_t = dx(dx(psi_t))*dy(dy(psi))+dx(dx(psi))*dy(dy(psi_t))-2*dx(dy(psi_t))*dx(dy(psi))

    prob_chi = d3.LBVP([chi, tau_chi], namespace=locals())
    prob_chi.add_equation("1/Ro*(lap(chi)-lap(lap(chi))) + tau_chi = -J(psi,lap(psi)) + lap(J(psi,h)) + 2*Ro*J_t - div(lap(psi)*grad(chi_old)) + lap(div(h*grad(chi_old)))")
    prob_chi.add_equation("integ(chi) = 0")

    # Solver
    solv_chi = prob_chi.build_solver()
    solv_chi.solve()

    # Fixed point algorithm ###########
    chi_old.change_scales(1); chi.change_scales(1)
    for step_i in range(100):
        
        chi_old['g'] = chi['g']

        solv_psit.solve()
        solv_chi.solve()
        
        chi_old.change_scales(1); chi.change_scales(1)
        
        jump = np.sqrt(np.mean( (chi_old['g']-chi['g'])**2) )
        print(jump)
        
        if jump < 1e-6*np.max(abs(chi_old['g'])):
            break
        
    # Save the data for SW ###########
    u_IC  = dist.Field(bases=(xbasis,ybasis))
    v_IC  = dist.Field(bases=(xbasis,ybasis))
    h_IC  = dist.Field(bases=(xbasis,ybasis))

    problem = d3.IVP([u_IC, v_IC, h_IC], namespace=locals())

    problem.add_equation("u_IC = -dy(psi)+dx(chi)")
    problem.add_equation("v_IC =  dx(psi)+dy(chi)")
    problem.add_equation("h_IC = h")

    ###
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = 10

    ###
    ICname = "SW_IC_%.2f_%d" %(Ro, rand_seed)
    ICname = ICname.replace(".", "d" ); ICname = ICname
    snapshots = solver.evaluator.add_file_handler(ICname, sim_dt=1e-10, max_writes=10)
    snapshots.add_task(-(-u_IC), name='u')
    snapshots.add_task(-(-v_IC), name='v')
    snapshots.add_task(-(-h_IC), name='h')

    ###
    solver.step(1e-10); solver.step(1e-10); solver.step(1e-10)

    # Save the data for QG+1 ###########
    q_IC  = dist.Field(bases=(xbasis,ybasis))

    zeta_IC = -dy(u_IC)+dx(v_IC)

    problem = d3.IVP([q_IC], namespace=locals())

    problem.add_equation("Ro*q_IC = (1+Ro*zeta_IC)/(1+Ro*h)-1")

    ###
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = 10

    ###
    ICname = "QGp1_IC_%.2f_%d" %(Ro, rand_seed)
    ICname = ICname.replace(".", "d" ); ICname = ICname
    snapshots = solver.evaluator.add_file_handler(ICname, sim_dt=1e-10, max_writes=10)
    snapshots.add_task(-(-q_IC), name='q')

    ###
    solver.step(1e-10); solver.step(1e-10); solver.step(1e-10)