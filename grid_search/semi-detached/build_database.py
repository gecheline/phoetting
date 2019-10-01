import phoebe
phoebe.mpi_off()
import numpy as np
import multiprocessing as mp 

def compute_lc(values):
    (q, r2, tratio, incl, ecc, per0) = values

    # b.set_value_all('atm', value='blackbody')
    # b.set_value_all('ld_mode', value='manual')
    b['q'] = q
    b['requiv@secondary'] = r2*b['value@sma@binary']
    b['teff@secondary'] = tratio*b['teff@primary'].value
    b['incl@binary'] = incl
    b['ecc'] = ecc
    b['per0'] = per0
    
    try:
        b.run_compute()
        return b['value@fluxes@model']
    except:
        return np.zeros(len(phases))


phases = np.linspace(-0.5,0.5,100)

b = phoebe.default_binary()
b.add_constraint('semidetached', 'primary')
b.add_dataset('lc', compute_phases=phases)

N=200000
qs = np.random.uniform(0.2,5.0,N)
rs2 = np.random.uniform(0.1,0.8,N)
tratios = np.random.uniform(0.5,2.0,N)
incls = np.random.uniform(10,90,N)
eccs = np.random.uniform(0.,0.6, N)
per0s = np.random.uniform(0.,360., N)

param_space = np.array([qs, rs2, tratios, incls, eccs, per0s]).T

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
np.savetxt('sdb_database.dat', lcs_space)
np.savetxt('sdb_database_params.dat', param_space)
