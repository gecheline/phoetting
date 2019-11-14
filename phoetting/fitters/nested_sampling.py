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

class NestedSampling(Fitter):

    def __init__(self, **kwargs):

        # for NestedSampling, you would preferably pass a bundle that's already set up
        # with observations and datasets and fixed parameters
        
        super(NestedSampling,self).__init__(**kwargs)

    
    def prior_transform(self, u):

        bounds = np.array(self.ranges)
    
        x = np.array(u)
        
        for i,bound in enumerate(bounds):
            x[i] = (bound[1] - bound[0])*u[i] + bound[0]
            
        return x


    def chi2_lc(self, bundle, ds, reduced=True):

        timesi = bundle['times@%s@dataset' % ds].value
        fluxes_model = bundle['fluxes@%s@model' % ds].interp_value(times=timesi)
        if reduced:
            scale = 1./(len(fluxes_model))
        else:
            scale = 1.
        
        return scale * (-0.5*np.sum((fluxes_model- \
                    bundle['value@fluxes@%s@dataset' % ds])**2 / \
                    bundle['value@sigmas@%s@dataset' % ds]**2 ))


    def chi2_rv(self, bundle, ds, component='primary', reduced = True):

        timesi = bundle['times@%s@%s@dataset' % (ds,component)].value
        rvs_model = bundle['rvs@model@%s@%s' % (ds,component)].interp_value(times=timesi)

        if reduced:
            scale = 1./(len(rvs_model))
        else:
            scale = 1.
        return scale*(-0.5*np.sum((rvs_model- \
                    bundle['value@%s@%s@dataset' % (ds, component)])**2 / \
                    bundle['value@sigmas@%s@%s@dataset' % (ds, component)]**2 ))


    def loglike(self, values, **kwargs):

        bundle = phoebe.load(self.bundle_file)
        try:
            bundle = self.set_params(bundle, self.params, values)
            bundle.run_compute()

            chi2 = []
            for ds in bundle.datasets:
                if ds[0:2] == 'lc':
                    chi2.append(self.chi2_lc(bundle, ds, **kwargs))
                elif ds[0:2] == 'rv':
                    # check if dataset exists for both componets or one
                    comps = set(bundle[ds+'@dataset'].components) - set(['binary'])
                    for c in comps:
                        chi2.append(self.chi2_rv(bundle, ds, component=c, **kwargs))
                else:
                    raise TypeError('Unsupported dataset type: %s' % ds)
            
            return np.sum(np.array(chi2))
                    
        except:
            return -1e12


    def run_dynesty(self, filename='dn_samples', parallel=True, saveiter=50, **kwargs):
        
        try:
            import dynesty as dn
        except:
            raise ImportError('dynesty cannot be imported')

        filename = kwargs.get('filename', self.bundle_file + '_dynesty')
 
        self.dynesty_file = filename
        ndim = len(self.params)
        self.__dn_filename = filename
        nlive = kwargs.pop('nlive', 500)

        if parallel:
            import multiprocessing as mp
            nproc = mp.cpu_count()
            pool = mp.Pool(nproc)
            
            sampler = dn.NestedSampler(self.loglike, self.prior_transform, ndim=ndim, nlive=nlive, pool=pool, queue_size=nproc, **kwargs)
        
        else:
            sampler = dn.NestedSampler(self.loglike, self.prior_transform, ndim=ndim, nlive=nlive, **kwargs)


        for result in sampler.sample(**kwargs):
            res = sampler.results
            if res['niter']%saveiter == 0:
                with open(self.dynesty_file, 'wb') as pfile:
                    pickle.dump(res, pfile)


    def dynesty_result(self):
        
        with open(self.dynesty_file, 'rb') as pfile:
            self.dn_result = pickle.load(pfile)
        
        return self.dn_result


            


        


