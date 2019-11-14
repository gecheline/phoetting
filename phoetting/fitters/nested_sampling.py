import phoebe
import numpy as np
from phoetting.fitters.main import Fitter 
import sys
import types
import pickle

if sys.version_info[0] < 3:
    import copy_reg as copyreg
else:
    import copyreg

def _pickle_method(m):
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)


def chi2_lc(bundle, ds, reduced=False):

    timesi = bundle['times@%s@dataset' % ds].value
    fluxes_model = bundle['fluxes@%s@model' % ds].interp_value(times=timesi)
    if reduced:
        scale = 1./(len(fluxes_model))
    else:
        scale = 1.
    
    return scale * (0.5*np.sum((fluxes_model- \
                bundle['value@fluxes@%s@dataset' % ds])**2 / \
                bundle['value@sigmas@%s@dataset' % ds]**2 ))


def chi2_rv(bundle, ds, component='primary', reduced = False):

    timesi = bundle['times@%s@%s@dataset' % (ds,component)].value
    rvs_model = bundle['rvs@model@%s@%s' % (ds,component)].interp_value(times=timesi)

    if reduced:
        scale = 1./(len(rvs_model))
    else:
        scale = 1.
    return scale*(0.5*np.sum((rvs_model- \
                bundle['value@%s@%s@dataset' % (ds, component)])**2 / \
                bundle['value@sigmas@%s@%s@dataset' % (ds, component)]**2 ))


def prior_transform(u, ranges):

    bounds = np.array(ranges)

    x = np.array(u)
    
    for i,bound in enumerate(bounds):
        x[i] = (bound[1] - bound[0])*u[i] + bound[0]
        
    return x


def loglike(values, bundle_file, params, reduce_chi2):

    bundle = phoebe.load(bundle_file)
    # try:
    bundle = Fitter.set_params(bundle, params, values)
    bundle.run_compute()

    logl = []
    for ds in bundle.datasets:
        if ds[0:2] == 'lc':
            logl.append(-chi2_lc(bundle, ds, reduced=reduce_chi2))
        elif ds[0:2] == 'rv':
            # check if dataset exists for both componets or one
            comps = set(bundle[ds+'@dataset'].components) - set(['binary'])
            for c in comps:
                logl.append(-chi2_rv(bundle, ds, component=c, reduced=reduce_chi2))
        else:
            raise TypeError('Unsupported dataset type: %s' % ds)
    print(values, np.sum(np.array(logl)))
    return np.sum(np.array(logl))
                
    # except:
    #     print(values, -1e12)
    #     return -1e12


class NestedSampling(Fitter):

    def __init__(self, **kwargs):

        # for NestedSampling, you would preferably pass a bundle that's already set up
        # with observations and datasets and fixed parameters
        
        super(NestedSampling,self).__init__(**kwargs)


    def run_dynesty(self, parallel=True, saveiter=50, **kwargs):
        
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
            
            logl_kwargs = {'bundle_file': self.bundle_file, 'params':self.params, 'reduce_chi2': reduce_chi2}
            
            sampler = dn.NestedSampler(loglike, prior_transform,
                                        logl_kwargs = logl_kwargs, ptform_kwargs={'ranges':self.ranges}, 
                                        ndim=ndim, nlive=nlive, pool=pool, queue_size=nproc, **kwargs)
        
        else:
            sampler = dn.NestedSampler(loglike, prior_transform, 
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