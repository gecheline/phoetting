import phoebe
import numpy as np
from phoetting.fitters.main import Fitter, compute_lc, compute_rvs
import sys
import types


if sys.version_info[0] < 3:
    import copy_reg as copyreg
else:
    import copyreg

def _pickle_method(m):
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)


class GridSearch(Fitter):

    def __init__(self, **kwargs):
        super(GridSearch,self).__init__(**kwargs)


    def clean_up_db(self, models, params):
        invalid = np.unique(np.argwhere((np.array(models) == 0.0))[:,0])
        models = np.delete(models, invalid, axis=0)
        params = np.delete(params, invalid, axis=0)

        return models, params


    def build_database(self, db_type='lc', N=100000, phase_params={'min':-0.5, 'max':0.5, 'len':100}, filename='', parallel=True):
        '''
        Parameters
        ----------
        N: int
            Number of models to compute in the database
        '''
        
        N = int(N)
        if db_type not in ['lc', 'rv']:
            raise ValueError('Database type %s not supported, can be one of [\'lc\',\'rv\']')

        phases = np.linspace(phase_params['min'],phase_params['max'],phase_params['len'])
        # add dataset to Bundle if it doesn't exist
        if len(self.bundle[db_type].twigs) == 0:
            #self.add_datasets(datasets=[db_type], compute_phases=[phases])
            self.bundle.add_dataset(db_type, compute_phases=phases)
            
        # make arrays to compute the parameters in
        param_space = np.random.uniform(self.ranges[:,0],self.ranges[:,1],(N,len(self.ranges)))

        # draw from arrays and compute model
        if db_type == 'lc':
            if parallel:
                import multiprocessing as mp
                from functools import partial
                numproc = mp.cpu_count() 
                print('Available processors: %s' % numproc)
                pool = mp.Pool(processes=numproc)
                results = pool.map(partial(compute_lc, params=self.params, bundle_file=self.bundle_file), param_space)
                lcs = np.array(results)
                
                pool.close()
                pool.join()
            
            else:
                lcs = np.zeros((N,len(phases)))
                for i, values in enumerate(param_space):
                    print('%i of %i' % (i, N))
                    lcs[i] = compute_lc(values=values, params=self.params, 
                                            bundle_file=self.bundle_file)
            
            lcs, param_space = self.clean_up_db(lcs,param_space)
            
            if len(filename) == 0:
                filename='lc_db'
            np.save(filename, lcs)
            np.save(filename+'_ps', param_space)
            self.db_file = filename
            
        elif db_type=='rv':
            if parallel:
                import multiprocessing as mp
                from functools import partial
                numproc = mp.cpu_count() 
                print('Available processors: %s' % numproc)
                pool = mp.Pool(processes=numproc)
                results = pool.map(partial(compute_rvs, ), param_space)
                rvs_1 = np.array(results[0])
                rvs_2 = np.array(results[1])
                
                pool.close()
                pool.join()
                
            else:
                rvs_1 = np.zeros((N,len(phases)))
                rvs_2 = np.zeros((N,len(phases)))
                for i, values in enumerate(param_space):
                    rvs_1[i], rvs_2[i] = compute_rvs(values, gscls=self)
            
            rvs_1, param_space = self.clean_up_db(rvs_1,param_space)
            rvs_2, _ = self.clean_up_db(rvs_2, param_space)
            
            if len(filename) == 0:
                filename='rv_db'
            np.save(filename+'_1', rvs_1)
            np.save(filename+'_2', rvs_2)
            np.save(filename+'_ps', param_space)
            self.db_file = filename

        else:
            raise TypeError(db_type)

    
    def distance_weighted_NN(self, database, test, database_params, test_params = [], plot=True, save_plot=True, **kwargs):
        
        from sklearn.neighbors import NearestNeighbors as NN
        
        def interp_nearest(lc_test, lc_db, nn=50):
            nbrs = NN(n_neighbors=nn).fit(lc_db)
            ds_0, inds_0 = nbrs.kneighbors(lc_test)
            ws = (1./ds_0)**2
            fs=(1./np.sum(ws,axis=1))
            weights=ws*fs[:,np.newaxis]
            return ds_0, inds_0, weights
        
        def plot_results(truths = [], skip=3, save=True):
            import matplotlib.pyplot as plt

            length = len(params_interp[:,0][::skip])
            print(length)
            fig, axes = plt.subplots(len(self.params), figsize=(12,24))
            
            for i in range(len(axes)):
                axes[i].scatter(range(length), params_interp[:,i][::skip], marker='o', c='r', s=20)
                axes[i].scatter(range(length), params_min[:,i][::skip], marker='o', c='g', s=10, alpha=0.5)
                axes[i].scatter(range(length), params_max[:,i][::skip], marker='o', c='b', s=10, alpha=0.5)
                if not len(truths)==0:
                    axes[i].scatter(range(length), truths[:,i][::skip], marker='x', c='k', s=50)
                for j in range(length):
                    axes[i].axvline(x=j, c='k', linestyle='--', alpha=0.2)
                axes[i].set_ylabel(self.params[i])
            
            if save:
                if 'filename' in kwargs.keys():
                    filename = kwargs['filename']
                else:
                    filename = self.db_file+'_results.png'
                fig.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()
            

        ds, inds, ws = interp_nearest(test, database, nn=10)
        params_interp = np.array([np.sum(database_params[:,i][inds]*ws,axis=1) for i in range(0, len(database_params[0]))]).T
        params_min = np.array([np.min(database_params[:,i][inds],axis=1) for i in range(0, len(database_params[0]))]).T
        params_max = np.array([np.max(database_params[:,i][inds],axis=1) for i in range(0, len(database_params[0]))]).T

        if plot:
            if 'skip' in kwargs.keys():
                skip = kwargs['skip']
            else:
                skip = 3
            plot_results(truths = test_params, skip=skip, save=save_plot)
  
