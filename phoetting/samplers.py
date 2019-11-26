import numpy as np 
from phoetting import utils
import phoebe
import pickle

#These live here so they can be pickleable for multiprocessing:
def prior_transform(u, ranges):

    bounds = np.array(ranges)

    x = np.array(u)
    
    for i,bound in enumerate(bounds):
        x[i] = (bound[1] - bound[0])*u[i] + bound[0]
        
    return x


def loglikelihood(values, bundle_file, params, reduce_chi2, fail_val = -1e12):

    bundle = phoebe.load(bundle_file)
    try:
        bundle = utils.set_params(bundle, params, values)
        bundle.run_compute()

        logl = []
        for ds in bundle.datasets:
            if ds[0:2] == 'lc':
                logl.append(-utils.chi2_lc(bundle, ds, reduced=reduce_chi2))
            elif ds[0:2] == 'rv':
                # check if dataset exists for both componets or one
                comps = set(bundle[ds+'@dataset'].components) - set(['binary'])
                for c in comps:
                    logl.append(-utils.chi2_rv(bundle, ds, component=c, reduced=reduce_chi2))
            else:
                raise TypeError('Unsupported dataset type: %s' % ds)
        return np.sum(np.array(logl))
                
    except:
        return fail_val


def logposterior(values, bundle_file, params, priors, reduce_chi2=False, fail_val = -np.inf, compute_logprior=True):

    if compute_logprior:
        try:
            import npdists as nd 
        except:
            raise ImportError('npdists cannot be imported - required for logprior computation')
        
        if len(priors) != len(values):
            raise ValueError('Mismatch between priors and values array size')

        logpri = nd.logp_from_dists(priors, values)
        if not np.isfinite(logp):
            return -np.inf 
    
    else:
        logpri = 0.

    loglike = loglikelihood(values, bundle_file, params, reduce_chi2, fail_val = fail_val)
    return loglike + logpri

    
def Sampler(object):

    def __init__(self, params={'incl@binary': [0.,90.]}, binary_type='detached', fixed_params={}, store_bundle=True, **kwargs):

        '''
        Initialize a Sampler instance with parameters to fit and fix in a phoebe Bundle.
        
        The provided dictionary of parameters is checked against existing twigs in the phoebe
        Bundle. If a match can be found, it's added to a list of good parameters and if not,
        to a list of bad parameters. If all parameters are bad, an Error is raised, otherwise
        a Warning states the bad parameters, while the good parameters are assigned to the Fitter.

        Parameters
        ----------
        params : dict
            A dictionary with phoebe-style twig parameter names and their corresponding ranges.
        binary_type : {'detached', 'semi-detached', 'contact}, optional
            The type of binary system to build if bundle not provided. Default is 'detached'.
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

        bundle = kwargs.get('bundle', utils.compute_bundle(binary_type=binary_type, **kwargs))

        good_ranges, good_params = utils.check_params(params=params, bundle=bundle)
        
        if bool(fixed_params):
            # user has provided parameters to be fixed in the bundle
            # need to follow phoebe twig logic and be in the default units

            for key in fixed_params.keys():
                try:
                    bundle.set_value_all(key, fixed_params[key])
                except:
                    raise Warning('%s is not a valid or explicit phoebe twig -- ignoring' % key)

        self.params = good_params 
        self.ranges = good_ranges
        self.fixed_params = fixed_params
        
        import string
        import random
        choices = string.ascii_letters+'0123456789'
        
        self.bundle_file = kwargs.get('bundle_file', 'tmpbundle'+''.join(random.choice(choices) for i in range(5)))
        bundle.save(self.bundle_file)
        if store_bundle:
            self.bundle = bundle


def Dynesty(Sampler):

    def __init__(self, **kwargs):

        # for NestedSampling, you would preferably pass a bundle that's already set up
        # with observations and datasets and fixed parameters
        
        super(Dynesty,self).__init__(**kwargs)


    def run_dynesty(self, parallel=True, saveiter=50, fail_val=-1e12, sample_args={}, **kwargs):
        
        try:
            import dynesty as dn
        except:
            raise ImportError('dynesty cannot be imported')

        self.dynesty_file = kwargs.pop('filename', self.bundle_file + '_dynesty')
        reduce_chi2 = kwargs.pop('reduce_chi2', False)

        ndim = len(self.params)
        nlive = kwargs.pop('nlive', 500)

        if parallel:
            import multiprocessing as mp
            nproc = mp.cpu_count()
            pool = mp.Pool(nproc)
            
            logl_kwargs = {'bundle_file': self.bundle_file, 'params':self.params, 
                            'reduce_chi2': reduce_chi2, 'fail_val':fail_val}
            
            sampler = dn.NestedSampler(loglikelihood, prior_transform,
                                        logl_kwargs = logl_kwargs, ptform_kwargs={'ranges':np.array(self.ranges)}, 
                                        ndim=ndim, nlive=nlive, pool=pool, queue_size=nproc, **kwargs)
        
        else:
            sampler = dn.NestedSampler(loglikelihood, prior_transform, 
                                        logl_kwargs = logl_kwargs, ptform_kwargs={'ranges': self.ranges}, 
                                        ndim=ndim, nlive=nlive, **kwargs)


        sargs = {}
        sargs['maxiter']=kwargs.pop('maxiter', None) 
        sargs['maxcall']=kwargs.pop('maxcall', None) 
        sargs['dlogz']=kwargs.pop('dlogz', 0.01) 
        sargs['logl_max']=kwargs.pop('logl_max', np.inf) 
        sargs['n_effective']=kwargs.pop('n_effective',np.inf) 
        sargs['add_live']=kwargs.pop('add_live', True) 
        sargs['save_bounds']=kwargs.pop('save_bounds', True) 
        sargs['save_samples']=kwargs.pop('save_samples',True)

        for result in sampler.sample(**sargs):
            res = sampler.results
            # if res['niter']%saveiter == 0:
            with open(self.dynesty_file, 'wb') as pfile:
                print('Saving results to %s...' % self.dynesty_file)
                pickle.dump(res, pfile)


    def dynesty_result(self):
        
        with open(self.dynesty_file, 'rb') as pfile:
            self.dn_result = pickle.load(pfile)
        
        return self.dn_result


def Emcee(Sampler):

    def __init__(self, nwalkers=128, niter=10000, priors = [], **kwargs):

        # for Emcee, you would preferably pass a bundle that's already set up
        # with observations and datasets and fixed parameters
        
        super(Emcee,self).__init__(**kwargs)
        self.nwalkers = nwalkers
        self.niter = niter

        # in Emcee the parameter "ranges" should be npdists
        # if they've been supplied as bounds, we need to "convert" them to npdists
        try:
            import npdists as nd
        except:
            raise ImportError('cannot import npdists - needed for prior distributions')
        
        for i,dist in enumerate(self.ranges):

            self.init_dists = []
            if not nd.is_distribution(dist):
                if isinstance(dist, ([list,np.ndarray])):
                    self.init_dists.append(nd.Uniform(low=dist[0], high=dist[1]))
                else:
                    raise ValueError('Cannot interpret parameter distributions or ranges,\
                                    provide a list/array of ranges of npdists distributions')
            else:
                self.init_dists.append(dist)
            
            # we don't need ranges here so remove 
        delattr(self, 'ranges')

        # check the same for the priors
        for i,pdist in enumerate(priors):
            
            self.priors = []
            if not isinstance(pdist, nd.BaseDistribution):
                if isinstance(dist, ([list,np.ndarray])):
                    self.priors.append(nd.Uniform(low=pdist[0], high=pdist[1]))
                else:
                    raise ValueError('Cannot interpret parameter distributions or ranges,\
                                    provide a list/array of ranges of npdists distributions')
            else:
                self.priors.append(pdist)


    def init_sample(self):
        '''
        Parameters
        ----------
        init_dists: list of npdists distributions
            The distributions to sample the intial position of the walkers from
        '''

        return np.array([self.init_dist.sample(self.nwalkers) for init_dist in p]).T

    

    def run_emcee(self, parallel=True, ensembler_args={}, sample_args = {}, **kwargs):
        try:
            import emcee
        except:
            raise ImportError('emcee cannot be imported')
        
        # pop and/or set EnsembleSampler kwargs
        esargs = {}
        esargs['nwalkers'] = self.nwalkers
        esargs['dim'] = len(self.params), 
        esargs['log_prob_fn'] = loglikelihood
        esargs['a']=kwargs.pop('a', None), 
        esargs['pool']=kwargs.pop('pool', None)
        esargs['moves']=kwargs.pop('moves', None)
        esargs['args']=None
        esargs['kwargs']={'bundle_file':self.bundle_file, 'params':self.params, 'priors':self.priors,
                'reduce_chi2': kwargs.pop('reduce_chi2', False), 'fail_val':kwargs.pop('fail_val', -np.inf), 
                'compute_logprior': kwargs.pop('compute_logprior', True)}

        esargs['live_dangerously']=kwargs.pop('live_dangerously', None)
        esargs['runtime_sortingfn']=kwargs.pop('runtime_sortingfn', None)
        
        if parallel:
            import multiprocessing as mp 
            nproc = mp.cpu_count()
            esargs['threads'] = nproc
            sampler = emcee.EnsembleSampler(**esargs)
        
        else:
            esargs['threads'] = None
            sampler = emcee.EnsembleSampler(**esargs)

        #sample kwargs
        sargs = {}
        sargs['p0'] = self.init_sample()
        sargs['log_prob0'] = kwargs.pop('log_prob0', None)
        sargs['rstate0'] = kwargs.pop('rstate0', None)
        sargs['blobs0'] = kwargs.pop('blobls0', None)
        sargs['iterations'] = self.niter
        sargs['thin'] = kwargs.pop('thin', 1)
        sargs['store'] = kwargs.pop('store', True)
        sargs['progress'] = kwargs.pop('progress', False)
        
        positions = []
        logps = []
        filename = kwargs.pop('filename', self.bundle_file+'_emcee')
        for result in sampler.sample(**sargs):
            position = result[0]
            f = open(filename, "a")
            for k in range(position.shape[0]):
                f.write("%d %s %f\n" % (k, " ".join(['%.12f' % i for i in position[k]]), result[1][k]))
            f.close()