import numpy as np 
import phoebe
from ipywidgets import *
from phoetting import utils

class ParamsPlotWidget(object):
    
    def __init__(self, names, twigs, ranges, bundle):
        
        self.names = names
        self.twigs = twigs
        self.ranges = ranges
        self.bundle = bundle

    def build_layout(self):

        # iterate over names and ranges and build and link slider/text
        # add to list of sliders/texts
        # group in HBoxes and VBoxes
        return None

    @staticmethod
    def make_slider_text(value, minv, maxv, step, description):
        slider = FloatSlider(
            value=value,
            min=minv,
            max=maxv,
            step=step,
            description=description,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False,
            readout_format='.5f',
            layout=Layout(width='200px')
        )

        text = FloatText(
            value=value,
            #description=description,
            disabled=False,
            layout=Layout(width='85px'),
        )
        return slider, text

    # @staticmethod
    # def update_vals_in_bundle(period=period, rsum=rsum, rratio=rratio, incl=incl,
    #                     esinw=eccen*np.sin(omega), ecosw=eccen*np.cos(omega), sma=sma, q=1.0, tratio=tratio,
    #                     pblum=12, lc=True, rv=False):
    #     [period, rsum, rratio, incl, esinw, ecosw, tratio, sma, q, pblum, vgamma] = [s.value for s in sliders]
        
    #     b['period@binary'] = period
    #     b['incl@binary'] = incl
    #     b['sma@binary'] = sma
    #     b['esinw'] = esinw
    #     b['ecosw'] = ecosw
    #     b['teff@secondary'] = tratio*b['value@teff@primary']
    #     b['requiv@primary'] = b['value@sma@binary']*rsum/(1.+rratio)
    #     b['requiv@secondary'] = b['value@sma@binary']*rsum*rratio/(1.+rratio)
    #     b['q'] = q
    #     b['pblum@primary'] = pblum
    #     b['vgamma'] = vgamma

    #     b.run_compute(compute='legacy01')#, ntriangles=500)

    @staticmethod
    def link(self, slider, text):
        widgetLink = jslink((slider, 'value'), (text, 'value'))

    # @staticmethod
    # def on_lc_button_clicked(button):
        
    #     for s in sliders:
    #         s.observe(update_vals_in_bundle, 'value')
        
    #     with plot:
    #         clear_output()
    #         plt.figure(figsize=(10,5))
    #         plt.plot(b.to_phase(b['value@times@lc@dataset']), b['fluxes@lc@model'].interp_value(times=lc[:,0]), '.', c='#008699')
    #         plt.scatter(b.to_phase(b['value@times@lc@dataset']), b['value@fluxes@lc@dataset'], c='#3f3535')
    #         plt.xlabel('Time (JD)')
    #         plt.ylabel('Flux (norm)')
    # #         plt.xlim([1730,1750])
    #         plt.show()

    # @staticmethod
    # def on_rv_button_clicked(button):

    #     for s in sliders:
    #         s.observe(update_vals_in_bundle, 'value')
    #     with plot:
    #         clear_output()
    #         plt.figure(figsize=(10,5))
    #         plt.plot(b.to_phase(b['value@times@rv@primary@model']), b['value@rvs@rv@primary@model'], '.', c='#008699')
    #         plt.scatter(b.to_phase(b['value@times@rv@primary@dataset']), b['value@rvs@rv@primary@dataset'], c='#3f3535')
    #         plt.xlabel('Time (JD)')
    #         plt.ylabel('RVs (km/s)')
    #         plt.show()

    # def display(self):
    #     return None