import phoebe
import numpy as np
from phoetting.fitters.main import Fitter 
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


    def chi2_lc(self, ds, reduced=True):

        timesi = self.bundle['times@%s@dataset' % ds].value
        fluxes_model = self.bundle['fluxes@%s@model' % ds].interp_value(times=timesi)
        if reduced:
            scale = 1./(len(fluxes_model))
        else:
            scale = 1.
        
        return scale * (-0.5*np.sum((fluxes_model- \
                    self.bundle['value@fluxes@%s@dataset' % ds])**2 / \
                    self.bundle['value@sigmas@%s@dataset' % ds]**2 ))


    def chi2_rv(self, ds, component='primary', reduced = True):

        timesi = self.bundle['times@%s@%s@dataset' % (ds,component)].value
        rvs_model = self.bundle['rvs@model@%s@%s' % (ds,component)].interp_value(times=timesi)

        if reduced:
            scale = 1./(len(rvs_model))
        else:
            scale = 1.
        return scale*(-0.5*np.sum((rvs_model- \
                    self.bundle['value@%s@%s@dataset' % (ds, component)])**2 / \
                    self.bundle['value@sigmas@%s@%s@dataset' % (ds, component)]**2 ))


    def loglike(self, values, **kwargs):

        try:
            self.set_params(values)
            self.bundle.run_compute()

            chi2 = []
            for ds in self.bundle.datasets:
                if ds[0:2] == 'lc':
                    chi2.append(self.chi2_lc(ds, **kwargs))
                elif ds[0:2] == 'rv':
                    # check if dataset exists for both componets or one
                    comps = set(self.bundle[ds+'@dataset'].components) - set(['binary'])
                    for c in comps:
                        chi2.append(self.chi2_rv(ds, component=c, **kwargs))
                else:
                    raise TypeError('Unsupported dataset type: %s' % ds)
            
            chi2 = np.array(chi2)
            return np.sum(chi2)
                    
        except:
            return -1e12


    def run_dynesty(self, filename='dn_samples', **kwargs):
        
        try:
            import dynesty as dn
        except:
            raise ImportError('dynesty cannot be imported')
        
        import multiprocessing as mp

        ndim = len(self.params)
        nproc = mp.cpu_count()
        pool = mp.Pool(nproc)
        self.__dn_filename = filename

        sampler = dn.NestedSampler(self.loglike, self.prior_transform, ndim, pool=pool, queue_size=nproc, **kwargs)

        # open up three files: samples_u, samples and sampler_params to write output in 
        for result in sampler.sample():

            # internally sampler saves the results as sampler.results
            # either save the last arg of results of dump after several iterations

            (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, 
                    h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = result


    def rejoin_dynesty_files(self):
        
        def make_dynesty_dict(samples, samples_u, params):
            ddict = {}
            ddict['samples'] = samples
            ddict['samples_u'] = samples_u
            
            for key in params.dtype.fields.keys():
                ddict[key] = params[key]
            
            return ddict

        samples = np.loadtxt(self.__dn_filename+'.dat', delimiter=',')
        samples_u = np.loadtxt(self.__dn_filename+'_u.dat', delimiter=',')
        params = np.loadtxt(self.__dn_filename + '_params.dat', delimiter=',')

        self.dn_result = make_dynesty_dict(samples, samples_u, params)
        return self.dn_result

    
    def plot_results(self, plotter='dynesty', save=True, savefile='dn_results.png', **kwargs):

        if not hasattr(self, 'dn_result'):
            self.rejoin_dynesty_files()
        
        if plotter == 'dynesty':
            import matplotlib.pyplot as plt
            from dynesty import plotting as dyplot
            cfig, caxes = dyplot.cornerplot(self.dn_result, labels=self.params, **kwargs)

            if save:
                cfig.savefig(savefile, dpi=300)
                plt.close()
            else:
                plt.show()
        
        else:
            try:
                import corner
            except:
                raise ImportError('Cannot import module corner (default if not plotter=\'dynesty\'')
            
            fig = corner.corner(self.dn_result['samples'], labels=self.params, quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, **kwargs)
            
            if save:
                fig.savefig(savefile, dpi=300)
                plt.close()
            else:
                plt.show()
            


        


