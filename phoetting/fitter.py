import phoebe
import numpy as np 
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

class Fitter(object):

    def __init__(self, params={'incl@binary': [0.,90.]}, bundle = None, binary_type='detached', fixed_params={}, **kwargs):

        '''
        Initialize a Fitter instance with parameters to fit and fix in a phoebe Bundle.
        
        The provided dictionary of parameters is checked against existing twigs in the phoebe
        Bundle. If a match can be found, it's added to a list of good parameters and if not,
        to a list of bad parameters. If all parameters are bad, an Error is raised, otherwise
        a Warning states the bad parameters, while the good parameters are assigned to the Fitter.

        Parameters
        ----------
        params : dict
            A dictionary with phoebe-style twig parameter names and their corresponding ranges.
        bundle : phoebe.Bundle instance, optional
            The phoebe Bundle to work with if available. The default value is None which
            creates a new Bundle instance.
        binary_type : {'detached', 'semi-detached', 'contact}, optional
            The type of binary system to build if bundle==None. Default is 'detached'.
        fixed_params: dict, optional
            A dictionary of parameters with known values to be fixed.
  

        Keyword Arguments
        -----------------
        flip_constraints: dict
            A dictionary of parameters to flip in format: 
            {'constrained parameter': new parameter to constrain}
        component : {'primary', 'secondary'}
          Only valid if binary_type='semi-detached' and determines which star
          the semi-detached constraint is applied to.
        flip_ff: bool
            Only valid if binary_type='contact', determines whether to flip the
            fillout factor constraint. Default is True if not provided.

        '''

        if bundle==None:
            if binary_type in ['detached', 'd', 'D', 'db', 'DB']:
                bundle = phoebe.default_binary()

            if binary_type in ['semi-detached', 'semidetached', 'sd', 'SD', 'sdb', 'SDB']:
                bundle = phoebe.default_binary()
                # check if user has provided component for sd constraint
                if 'component' in kwargs.keys():
                    sd_comp = kwargs['component']
                else:
                    sd_comp = 'primary'
                bundle.add_constraint('semi-detached', sd_comp)
            
            if binary_type in ['contact', 'overcontact', 'c', 'oc', 'C', 'OC', 'cb', 'CB', 'ocb', 'OCB']:
                bundle = phoebe.default_binary(contact_binary=True)
                # by default, we'll flip the constraint for requiv and ff
                # but check if user doesn't want to
                if 'flip_ff' in kwargs.keys():
                    flip_ff = kwargs['flip_ff']
                else:
                    flip_ff = True
                    
                if flip_ff:
                    bundle.flip_constraint('pot', solve_for='requiv@primary')
                    bundle.flip_constraint('fillout_factor', solve_for='pot')

            if 'flip_constraints' in kwargs.keys():
                for key in kwargs['flip_constraints'].keys():
                    bundle.flip_constraint(key, solve_for=kwargs['flip_constraints'][key])

            if bool(fixed_params):
                # user has provided parameters to be fixed in the bundle
                # need to follow phoebe twig logic and be in the default units

                for i,key in enumerate(fixed_params.keys()):
                    try:
                        bundle[key] = fixed_params[key]
                    except:
                        raise Warning('%s is not a valid or explicit phoebe twig -- ignoring' % key)


        # check if parameters make sense, exist in Bundle or could be matched to Bundle
        good_params = []
        good_ranges = []
        bad_params = []

        for pkey in params.keys():
            try:
                bundle.get_parameter(pkey)
                good_params.append(pkey)
                good_ranges.append(params[pkey])
            except:
                bad_params.append(pkey)

        if len(good_params) == 0:
            raise KeyError('No good phoebe twigs found in provided parameters. \
            Rename to match unique phoebe twigs!')

        elif len(good_params) < len(params):
            raise Warning('%s not valid or explicit phoebe twigs -- ignoring!' % bad_params)

        else:
            print('All parameter names match phoebe twigs! Good job!')

        self.params = good_params 
        self.ranges = np.array(good_ranges)
        self.bundle = bundle

    
    def set_params(self, values):

        '''
        Set the Bundle parameter values.

        Parameters
        ----------
        values: array-like
            An array of length = self.params that specifies the values.

        
        Raises
        ------
        ValueError 
            if the length of the values array doesn't match the number of parameters in
            the Fitter instance.
        '''
        
        # check if provided value array matches the size of parameters to fit
        if len(values) == len(self.params):
            for i,pkey in enumerate(self.params):
                self.bundle[pkey] = values[i]
        else:
            raise ValueError('Size mismatch between value array (%s) and parameters array (%s)' % (len(values),len(self.params)))
        

    def add_datasets(self, datasets=['lc'], compute_phases=[], compute_times=[], npoints=100):

        '''
        Add an 'lc' or/and 'rv' dataset to the Bundle with an array of compute times/phases.

        Parameters
        ----------
        datasets: list
            A list of datasets to add to the Bundle. Available options: {'lc', 'rv'}
        compute_phases: list
            A list of arrays specifying the phases to compute the corresponding dataset in.
        compute_times: list
            A list of arrays specifying the times to compute the corresponding dataset in.
        npoints: int, list, optional
            If no compute_phases or compute_times provided, the number of points in which
            to compute them in.
        '''

        # so far it looks like this is unnecessarily complicated and it's better
        # to leave it up to the user or specialized fitters to handle the datasets
        self.datasets = datasets

        if not isinstance(compute_phases, list) or not isinstance(compute_times, list):
            raise TypeError('Compute phases and times need to lists of arrays \
            matching the length of the datasets list')

        if len(compute_times) == 0:
            if len(compute_phases) == 0:
                if isinstance(npoints, (list,np.array)):
                    if len(npoints)!=len(datasets):
                        raise ValueError('Size mismatch between datasets list and npoints list')
                else:
                    npoints_arr = []
                    for i in range(len(datasets)):
                        npoints_arr.append(npoints)  
                    npoints = npoints_arr

                for i in range(len(npoints)):
                    compute_phases.append(np.linspace(-0.5,0.5,npoints[i]))
            
            else:
                if len(datasets) > 1:  

                    if len(compute_phases)==1:
                        compute_phases_arr = []
                        for i in range(len(datasets)):
                            compute_phases_arr.append(compute_phases[0])
                        compute_phases = compute_phases_arr
                    
                    else:
                        if len(compute_phases) != len(datasets):
                            raise ValueError('Size mismatch between datasets list and phases list')
                else:
                    if len(compute_phases) != len(datasets):
                        raise ValueError('Size mismatch between datasets list and phases list')

            for i,dataset in enumerate(datasets):
                try:
                    self.bundle.add_dataset(datasets, compute_phases=compute_phases[i])
                except Exception as e: 
                    print(e)


        else:
            if len(datasets)>1:
                if len(compute_times) == 1:
                    compute_times_arr = []
                    for i in range(len(datasets)):
                        compute_times_arr.append(compute_times[0])
                    compute_times=compute_times_arr
                else:
                    if len(datasets) != len(compute_times):
                        raise ValueError('Size mismatch between datasets list and times list')
            
            else:
                if len(compute_times) != len(datasets):
                    raise ValueError('Size mismatch between datasets list and times list')
            
            for i,dataset in enumerate(datasets):
                try:
                    self.bundle.add_dataset(datasets, compute_times=compute_times[i])
                except Exception as e: 
                    print(e)


class GridSearch(Fitter):

    def __init__(self, **kwargs):
        super(GridSearch,self).__init__(**kwargs)


    def compute_lc(self, values):
        # set values of all the parameters and return computed model
        
        if len(values) != len(self.params):
            raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(self.params)))
        
        self.set_params(values)
   
        try:
            self.bundle.run_compute()
            return self.bundle['value@fluxes@model']
        except:
            return np.zeros(len(self.bundle['compute_phases']))


    def compute_rvs(self,values):

                
        if len(values) != len(self.params):
            raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(self.params)))
        
        self.set_params(values)
   
        try:
            self.bundle.run_compute()
            return [self.bundle['value@rvs@primary@model'], self.bundle['value@rvs@secondary@model']]

        except:
            return [np.zeros(len(self.bundle['compute_phases'])),np.zeros(len(self.bundle['compute_phases']))]


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
                numproc = mp.cpu_count() 
                print('Available processors: %s' % numproc)
                pool = mp.Pool(processes=numproc)
                results = pool.map(self.compute_lc, param_space)
                lcs = np.array(results)
                
                pool.close()
                pool.join()
            
            else:
                lcs = np.zeros((N,len(phases)))
                for i, values in enumerate(param_space):
                    lcs[i] = self.compute_lc(values)
            
            lcs, param_space = self.clean_up_db(lcs,param_space)
            
            if len(filename) == 0:
                filename='lc_db'
            np.save(filename, lcs)
            np.save(filename+'_ps', param_space)
            self.db_file = filename
            
        elif db_type=='rv':
            if parallel:
                import multiprocessing as mp
                numproc = mp.cpu_count() 
                print('Available processors: %s' % numproc)
                pool = mp.Pool(processes=numproc)
                results = pool.map(self.compute_lc, param_space)
                rvs_1 = np.array(results[0])
                rvs_2 = np.array(results[1])
                
                pool.close()
                pool.join()
                
            else:
                rvs_1 = np.zeros((N,len(phases)))
                rvs_2 = np.zeros((N,len(phases)))
                for i, values in enumerate(param_space):
                    rvs_1[i], rvs_2[i] = self.compute_rvs(values)
            
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
        
class NestedSampling(Fitter):

    def __init__(self, code='dynesty'):
        return None


class MCMC(Fitter):

    def __init__(self, code='emcee'):
        return None





    



    

