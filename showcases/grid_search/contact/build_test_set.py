import phoebe
phoebe.mpi_off()
import numpy as np
import multiprocessing as mp 

def compute_lc(values):
    q = values[3]
    ff = values[2]
    tratio = values[1]
    incl = values[0]

    b = phoebe.default_binary(contact_binary = True)
    b.add_dataset('lc', compute_phases=phases)
    b['pblum_mode'] = 'pbflux'
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    # b.set_value_all('atm', value='blackbody')
    # b.set_value_all('ld_mode', value='manual')
    b['q'] = q
    b['fillout_factor'] = ff 
    b['teff@secondary'] = tratio*b['teff@primary'].value
    b['incl@binary'] = incl
    
    try:
        b.run_compute()
        return b['value@fluxes@model']
    except:
        return np.zeros(len(phases))


phases = np.linspace(-0.5,0.5,100)

N=300
qs = np.random.uniform(0.2,1.0,N)
ffs = np.random.uniform(0.1,1.0,N)
tratios = np.random.uniform(0.5,2.0,N)
incls = np.random.uniform(10,90,N)

param_space = np.array([incls,tratios,ffs,qs]).T
lcs_space = np.zeros((N, len(phases)))

numproc = mp.cpu_count() 
print('Available processors: %s' % numproc)
pool = mp.Pool(processes=numproc)

lcs_space = np.array(pool.map(compute_lc, param_space))

pool.close()
pool.join()

# clean up
invalid = np.unique(np.argwhere((np.array(lcs_space) == 0.0))[:,0])
lcs_space = np.delete(lcs_space, invalid, axis=0)
param_space = np.delete(param_space, invalid, axis=0)

# save files
np.savetxt('cb_testset.dat', lcs_space)
np.savetxt('cb_testset_params.dat', param_space)
