import phoebe
import numpy as np

def compute_bundle(binary_type='detached', **kwargs):

    '''
    Create a phoebe Bundle object and adjust parameters and constraints.

    Parameters
    ----------
    binary_type: str
        Specifies whether to build a detached, semi-detached or contact system.
        Options: ('detached', 'semi-detached', 'contact')

    
    Returns
    -------
    bundle: phoebe.Bundle 
        The computed bundle object.
    '''
        
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


def check_params(params, bundle):

    '''
    Set the Bundle parameter values.

    Parameters
    ----------
    params: dict
        A dictionary of parameters and corresponding ranges/values.
    bundle: phoebe.Bundle
        The phoebe bundle object that is provided by the user or computed within the code.

    
    Raises
    ------
    Warning
        if some parameter names don't have matching Bundle twigs and will be ignored.
    KeyError 
        if none of the parameter names match Bundle twigs.
    '''

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
    
    if len(values) == len(params):
        for i,pkey in enumerate(params):
            bundle[pkey] = values[i]
    else:
        raise ValueError('Size mismatch between value array (%s) and parameters array (%s)' % (len(values),len(params)))
    
    return bundle


def compute_lc(values, params, bundle_file=None, bundle=None):

    '''
    Computes the light curve given a set of parameter values.

    Parameters
    ----------
    values: array-like
    params: array-like
    bundle_file: str
    bundle: phoebe.Bundle

    Returns
    -------
    bundle: phoebe.Bundle


    Raises
    ------
    '''

    if bundle is None:
        if bundle_file is None:
            raise ValueError('Must provide either bundle of bundle_file to load from!')
        else:
            bundle = phoebe.load(bundle_file)

    if len(values) != len(params):
        raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(params)))
    
    bundle = set_params(bundle, params, values)

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


def compute_rvs(values, params, bundle_file=None, bundle=None):
    '''
    Computes the radial velocity curves given a set of parameter values.
    '''

    if bundle is None:
        if bundle_file is None:
            raise ValueError('Must provide either bundle of bundle_file to load from!')
        else:
            bundle = phoebe.load(bundle_file)

    if len(values) != len(params):
        raise ValueError('Shape mismatch between values array (%i) and parameters (%i)!' % (len(values), len(params)))
    
    bundle = set_params(bundle, params, values)

    try:
        bundle.run_compute()
        return [bundle['value@rvs@primary@model'], bundle['value@rvs@secondary@model']]

    except:
        return [np.zeros(len(bundle['compute_phases'])),np.zeros(len(bundle['compute_phases']))]


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