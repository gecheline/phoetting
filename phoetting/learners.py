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


def compute_lc(values, params, bundle_file):
    print('Computing light curve for %s' % values)
    bundle = phoebe.load(bundle_file)
    if len(values) != len(params):
        raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(params)))
    
    bundle = PhoebeLearner.set_params(bundle, params, values)

    try:
        bundle.run_compute()
        return bundle['value@fluxes@model']
    except:
        try:
            bundle.set_value_all('atm', 'blackbody')
            bundle.set_value_all('ld_mode', 'manual')
            bundle.run_compute()
            return bundle['value@fluxes@model']
        except:
            return np.zeros(len(bundle['value@compute_phases']))


def compute_rvs(values, params, bundle_file):

    bundle = phoebe.load(bundle_file)
    if len(values) != len(params):
        raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(params)))
    
    bundle = PhoebeLearner.set_params(bundle, params, values)

    try:
        bundle.run_compute()
        return [bundle['value@rvs@primary@model'], bundle['value@rvs@secondary@model']]

    except:
        return [np.zeros(len(bundle['compute_phases'])),np.zeros(len(bundle['compute_phases']))]

class PhoebeLearner(object):

    def __init__(self, params={'incl@binary': [0.,90.]}, binary_type='detached', fixed_params={}, store_bundle=True, **kwargs):

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

        bundle = kwargs.get('bundle', self.compute_bundle(binary_type=binary_type, **kwargs))

        good_ranges, good_params = self.check_params(params=params, bundle=bundle)
        
        if bool(fixed_params):
            # user has provided parameters to be fixed in the bundle
            # need to follow phoebe twig logic and be in the default units

            for key in fixed_params.keys():
                try:
                    bundle.set_value_all(key, fixed_params[key])
                except:
                    raise Warning('%s is not a valid or explicit phoebe twig -- ignoring' % key)

        self.params = good_params 
        self.ranges = np.array(good_ranges)
        self.fixed_params = fixed_params
        
        import string
        import random
        choices = string.ascii_letters+'0123456789'
        
        self.bundle_file = kwargs.get('bundle_file', 'tmpbundle'+''.join(random.choice(choices) for i in range(8)))
        bundle.save(self.bundle_file)
        if store_bundle:
            self.bundle = bundle


    @staticmethod
    def compute_bundle(binary_type='detached', **kwargs):
        
        if binary_type in ['detached', 'd', 'D', 'db', 'DB']:
            bundle = phoebe.default_binary()

        if binary_type in ['semi-detached', 'semidetached', 'sd', 'SD', 'sdb', 'SDB']:
            bundle = phoebe.default_binary()
            # check if user has provided component for sd constraint
            sd_comp = kwargs.get('component', 'primary')
            bundle.add_constraint('semi-detached', sd_comp)
        
        if binary_type in ['contact', 'overcontact', 'c', 'oc', 'C', 'OC', 'cb', 'CB', 'ocb', 'OCB']:
            bundle = phoebe.default_binary(contact_binary=True)
            # by default, we'll flip the constraint for requiv and ff
            # but check if user doesn't want to
            flip_ff = kwargs.get('flip_ff', True)

            if flip_ff:
                bundle.flip_constraint('pot', solve_for='requiv@primary')
                bundle.flip_constraint('fillout_factor', solve_for='pot')

        if 'flip_constraints' in kwargs.keys():
            for key in kwargs['flip_constraints'].keys():
                bundle.flip_constraint(key, solve_for=kwargs['flip_constraints'][key])
        
        return bundle


    @staticmethod
    def check_params(params={}, bundle=None):

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

        return good_ranges, good_params


    @staticmethod
    def set_params(bundle, params, values):

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
        if len(values) == len(params):
            for i,pkey in enumerate(params):
                bundle[pkey] = values[i]
        else:
            raise ValueError('Size mismatch between value array (%s) and parameters array (%s)' % (len(values),len(params)))
        
        return bundle


class Database(PhoebeLearner):

    def __init__(self, **kwargs):
        super(Database,self).__init__(**kwargs)


    def clean_up_db(self, models, params):
        '''
        Removes failed light curves and their associated parameters from the set.
        '''
        invalid = np.unique(np.argwhere((np.array(models) == 0.0))[:,0])
        models = np.delete(models, invalid, axis=0)
        params = np.delete(params, invalid, axis=0)

        return models, params


    def build_database(self, db_type='lc', N=100000, phase_params={'min':-0.5, 'max':0.5, 'len':100}, filename='', parallel=True):
        '''


        Parameters
        ----------
        N: int
            Number of models to compute in the database.
        db_type: str
            Choose between light curves and radial velocity curves.
        phase_params: dict
            min/max/len of the phase array.
        filename: str
            Filename to save the library to.
        parallel: bool
            Toggle between serial/parallel computation.
        '''

        bundle = phoebe.load(self.bundle_file)

        N = int(N)
        if db_type not in ['lc', 'rv']:
            raise ValueError('Database type %s not supported, can be one of [\'lc\',\'rv\']')

        phases = np.linspace(phase_params['min'],phase_params['max'],phase_params['len'])
        # add dataset to Bundle if it doesn't exist
        if len(self.bundle[db_type].twigs) == 0:
            #self.add_datasets(datasets=[db_type], compute_phases=[phases])
            bundle.add_dataset(db_type, compute_phases=phases)
        bundle.save(self.bundle_file)    
            
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
            np.savetxt(filename+'_ps.csv', param_space, delimiter=',', header=",".join(self.params))
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
                    rvs_1[i], rvs_2[i] = compute_rvs(values=values, params=self.params, 
                                                        bundle_file = self.bundle_file)
            
            rvs_1, param_space = self.clean_up_db(rvs_1,param_space)
            rvs_2, _ = self.clean_up_db(rvs_2, param_space)
            
            if len(filename) == 0:
                filename='rv_db'
            np.save(filename+'_1', rvs_1)
            np.save(filename+'_2', rvs_2)
            np.savetxt(filename+'_ps.csv', param_space, delimiter=',', header=",".join(self.params))
            self.db_file = filename
            self.db_type = db_type

        else:
            raise TypeError(db_type)

    def compute_downprojection(self, algorithm='TSNE', algorithm_kwargs = {}, skip=1, store=True, comp=1):

        '''Algorithm has to be a method of sklearn.manifold'''

        def import_from(module, name):
            module = __import__(module, fromlist=[name])
            return getattr(module, name)

        try:
            mapping = import_from('sklearn.manifold', algorithm)
        except:
            import inspect
            import sklearn
            raise ImportError('%s not a method of sklearn.manifold, try one of %s' % (algorithm, inspect.getmembers(sklearn.manifold, predicate=inspect.isclass)))
        
        if self.db_type == 'lc':
            db = np.load(self.db_file)

        elif self.db_type == 'rv':
            db = np.load(self.db_file+comp)
        
        dbmap = mapping(**algorithm_kwargs).fit_transform(db[::skip])
        if store:
            self.downprojection = dbmap
        
        return dbmap

    
    def plot_downprojection(self, dbmap, colorparams=['incl@binary'], skip=1, show=True, **kwargs):

        import matplotlib.pyplot as plt
        ps = np.genfromtxt(self.db_file + '_ps', delimiter=',', names=True)

        figsize = kwargs.pop('figsize', (10,40))
        fig, axes = plt.subplots(len(colorparams), figsize=figsize)

        # this is only here to allow choice of axes for a 3D projection
        x = kwargs.pop('x', 0)
        y = kwargs.pop('y', 1)

        for i in range(len(axes)):
            cb = axes[i].scatter(dbmap[:,x], dbmap[:,y], c=ps[colorparams[i]][::skip])
            fig.colorbar(cb, ax = axes[i], label=colorparams[i])
        
        if show:
            plt.show()
        return fig


class NeighborsSearch(object):

    def __init__(self, dbfile = '', psfile = '', xfile = '', xpsfile = None):
        '''
        Parameters
        ----------
        dbfile: str
            Path to library file.
        psfile: str
            Path to parameter value file.
        xfile: str
            Path to light curve or dataset whose parameters are 
            to be estimated from the library.
        xpsfile: str
            Path to xfile true parameter values (if known).
        '''
        self.dbfile = dbfile
        self.psfile = psfile
        self.xfile = xfile
        self.xpsfile = xpsfile


    def load_files(self):
        db = np.load(self.dbfile)
        ps = np.loadtxt(self.psfile)
        x = np.load(self.xfile)
        return db, ps, x

    @staticmethod
    def find_nearest_neighbors(db, test, nn=10):

        from sklearn.neighbors import NearestNeighbors as NN
        nbrs = NN(n_neighbors=nn).fit(db)
        ds_0, inds_0 = nbrs.kneighbors(test)
        ws = (1./ds_0)**2
        fs=(1./np.sum(ws,axis=1))
        weights=ws*fs[:,np.newaxis]
        return ds_0, inds_0, weights


    def compute_dw_means(self, nn=10, plot=False, save=True, **kwargs):

        db, ps, x = self.load_files()
        db_ds, db_inds, db_ws = self.find_nearest_neighbors(db, x, nn=nn)
        ps_interp = np.array([np.sum(ps[:,i][db_inds]*db_ws,axis=1) for i in range(0, len(ps[0]))]).T
        ps_min = np.array([np.min(ps[:,i][db_inds],axis=1) for i in range(0, len(ps[0]))]).T
        ps_max = np.array([np.max(ps[:,i][db_inds],axis=1) for i in range(0, len(ps[0]))]).T

        if plot:
            import matplotlib.pyplot as plt

            if not self.xpsfile == None:
                xps = np.loadtxt(self.xpsfile)

            skip=kwargs.pop('skip', 3)
            labels = kwargs.pop('labels', range(len(ps[0])))
            figsize = kwargs.pop('figsize', (12,24))
            length = len(ps_interp[:,0][::skip])
            
            fig, axes = plt.subplots(len(ps[0]), figsize=figsize)
            for i in range(len(axes)):
                axes[i].scatter(range(length), ps_interp[:,i][::skip], marker='o', c='r', s=20)
                axes[i].scatter(range(length), ps_min[:,i][::skip], marker='o', c='g', s=10, alpha=0.5)
                axes[i].scatter(range(length), ps_max[:,i][::skip], marker='o', c='b', s=10, alpha=0.5)
                if not self.xpsfile == None:
                    axes[i].scatter(range(length), xps[:,i][::skip], marker='x', c='k', s=50)
                for j in range(length):
                    axes[i].axvline(x=j, c='k', linestyle='--', alpha=0.2)
                axes[i].set_ylabel(labels[i])

        db_results = {}
        db_results['dw_mean'] = ps_interp
        db_results['mins'] = ps_min
        db_results['maxs'] = ps_max

        if save:
            import pickle

            filename = kwargs.pop('filename', 'grid_resuls.p')
            pickle.dump(db_results, open(filename, "wb" ))

        return db_results


class NeuralNetwork(object):

    def __init__(self):
        return None