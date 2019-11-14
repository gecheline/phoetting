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
    
    bundle = Fitter.set_params(bundle, params, values)

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
    
    bundle = Fitter.set_params(bundle, params, values)

    try:
        bundle.run_compute()
        return [bundle['value@rvs@primary@model'], bundle['value@rvs@secondary@model']]

    except:
        return [np.zeros(len(bundle['compute_phases'])),np.zeros(len(bundle['compute_phases']))]


class Fitter(object):

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
        self.bundle_file = 'tmpbundle'+''.join(random.choice(choices) for i in range(8))
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


    # def add_datasets(self, datasets=['lc'], compute_phases=[], compute_times=[], npoints=100):

        # '''
        # Add an 'lc' or/and 'rv' dataset to the Bundle with an array of compute times/phases.

        # Parameters
        # ----------
        # datasets: list
        #     A list of datasets to add to the Bundle. Available options: {'lc', 'rv'}
        # compute_phases: list
        #     A list of arrays specifying the phases to compute the corresponding dataset in.
        # compute_times: list
        #     A list of arrays specifying the times to compute the corresponding dataset in.
        # npoints: int, list, optional
        #     If no compute_phases or compute_times provided, the number of points in which
        #     to compute them in.
        # '''

        # # so far it looks like this is unnecessarily complicated and it's better
        # # to leave it up to the user or specialized fitters to handle the datasets
        # self.datasets = datasets

        # if not isinstance(compute_phases, list) or not isinstance(compute_times, list):
        #     raise TypeError('Compute phases and times need to lists of arrays \
        #     matching the length of the datasets list')

        # if len(compute_times) == 0:
        #     if len(compute_phases) == 0:
        #         if isinstance(npoints, (list,np.array)):
        #             if len(npoints)!=len(datasets):
        #                 raise ValueError('Size mismatch between datasets list and npoints list')
        #         else:
        #             npoints_arr = []
        #             for i in range(len(datasets)):
        #                 npoints_arr.append(npoints)  
        #             npoints = npoints_arr

        #         for i in range(len(npoints)):
        #             compute_phases.append(np.linspace(-0.5,0.5,npoints[i]))
            
        #     else:
        #         if len(datasets) > 1:  

        #             if len(compute_phases)==1:
        #                 compute_phases_arr = []
        #                 for i in range(len(datasets)):
        #                     compute_phases_arr.append(compute_phases[0])
        #                 compute_phases = compute_phases_arr
                    
        #             else:
        #                 if len(compute_phases) != len(datasets):
        #                     raise ValueError('Size mismatch between datasets list and phases list')
        #         else:
        #             if len(compute_phases) != len(datasets):
        #                 raise ValueError('Size mismatch between datasets list and phases list')

        #     for i,dataset in enumerate(datasets):
        #         try:
        #             self.bundle.add_dataset(datasets, compute_phases=compute_phases[i])
        #         except Exception as e: 
        #             print(e)


        # else:
        #     if len(datasets)>1:
        #         if len(compute_times) == 1:
        #             compute_times_arr = []
        #             for i in range(len(datasets)):
        #                 compute_times_arr.append(compute_times[0])
        #             compute_times=compute_times_arr
        #         else:
        #             if len(datasets) != len(compute_times):
        #                 raise ValueError('Size mismatch between datasets list and times list')
            
        #     else:
        #         if len(compute_times) != len(datasets):
        #             raise ValueError('Size mismatch between datasets list and times list')
            
        #     for i,dataset in enumerate(datasets):
        #         try:
        #             self.bundle.add_dataset(datasets, compute_times=compute_times[i])
        #         except Exception as e: 
        #             print(e)