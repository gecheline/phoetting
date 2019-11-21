import numpy as np 
import phoebe
from ipywidgets import *
from IPython.display import display, clear_output
from phoetting import utils
import matplotlib.pyplot as plt

class ParamsPlotWidget(object):
    
    def __init__(self, bundle, params):

        '''
        Parameters
        ----------
        bundle: phoebe.Bundle object
            The bundle to work with in the widget
        params: dict
            Parameter names as phoebe twigs and their corresponding ranges
        '''

        self.twigs, self.ranges = utils.check_params(params, bundle)
        self.bundle = bundle
        self.build_layout()

    def build_layout(self):

        # iterate over names and ranges and build and link slider/text
        # add to list of sliders/texts
        # group in HBoxes and VBoxes
        self.sliders = []
        self.texts = []
        for name, box in zip(self.twigs, self.ranges):
            value = 0.5*(box[0]+box[1])
            step = (box[1]-box[0])/100.
            slider, text = self.make_slider_text(value,box[0],box[1],step,name)
            self.sliders.append(slider)
            self.texts.append(text)

        lc_button = Button(description="Plot LC")
        rv_button = Button(description="Plot RV")
        self.plot = Output()

        lc_button.on_click(self.on_lc_button_clicked)
        rv_button.on_click(self.on_rv_button_clicked)

        #TODO: add options for phoebe-backend and compute options (ntriangles, gridsize...)
        

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

        jslink((slider, 'value'), (text, 'value'))
        return slider, text


    def update_vals_in_bundle(self):

        for i in range(len(self.twigs)):
            self.bundle[self.twigs[i]] = self.sliders[i].value

        self.bundle.run_compute()#, ntriangles=500)


    def on_lc_button_clicked(self, button):
        
        for s in self.sliders:
            s.observe(self.update_vals_in_bundle, 'value')
        
        with self.plot:
            clear_output()
    
    #         plt.figure(figsize=(10,5))
    #         plt.plot(b.to_phase(b['value@times@lc@dataset']), b['fluxes@lc@model'].interp_value(times=lc[:,0]), '.', c='#008699')
    #         plt.scatter(b.to_phase(b['value@times@lc@dataset']), b['value@fluxes@lc@dataset'], c='#3f3535')
    #         plt.xlabel('Time (JD)')
    #         plt.ylabel('Flux (norm)')
    # #         plt.xlim([1730,1750])
    #         plt.show()


    def on_rv_button_clicked(self, button):

        for s in self.sliders:
            s.observe(self.update_vals_in_bundle, 'value')
        
        with self.plot:
            clear_output()
            # plt.figure(figsize=(10,5))
            # plt.plot(b.to_phase(b['value@times@rv@primary@model']), b['value@rvs@rv@primary@model'], '.', c='#008699')
            # plt.scatter(b.to_phase(b['value@times@rv@primary@dataset']), b['value@rvs@rv@primary@dataset'], c='#3f3535')
            # plt.xlabel('Time (JD)')
            # plt.ylabel('RVs (km/s)')
            # plt.show()

    # def display(self):
    #     return None