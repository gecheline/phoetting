import numpy as np 
from phoetting import utils

def Sampler(object):

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

        bundle = kwargs.get('bundle', utils.compute_bundle(binary_type=binary_type, **kwargs))

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

    def priors(self):
        return None 

    def loglikelihood(self):
        # figure out if this can live here with multiprocessing
        return None


def Dynesty(Sampler):

    def __init__(self):
        import dynesty
        return None 


def Emcee(Sampler):

    def __init__(self):
        import emcee
        return None
