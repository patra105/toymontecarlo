'''This class is for dataset manipulation consisting of either one list of data or two lists of correlated data.'''
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit

class DataSet:
    '''
    (c) Sean Poczos 2025

    Attributes:
    - _datalist_1: list | np.ndarray
    - _datalist_2: None | list | np.ndarray
    - _metadata: dict

    Public methods:
    + set_datalist(
                   datalist_num: int,
                   new_datalist: list<float>
                  ): None

    + get_datalist(
                   datalist_num: int
                  ): list<float>

    + print_datalist(
                     datalist_num: int
                    ): None

    + add_field(
                fieldname: str,
                value: str | int | float | list | bool
               ): None

    + get_metadata(): dict

    + print_metadata(): None

    + make_scatterplot(
                       fig_dict: dict,
                       figname: str,
                       axname: str,
                       y_data: np.ndarray,
                       x_data = None: np.ndarray,
                       overwrite = False: bool
                      ): None

    + make_1d_hist(
                   fig_dict: dict,
                   figname: str,
                   axname: str,
                   savepath: str,
                   data: np.ndarray,
                   x_min: float,
                   x_max: float,
                   bin_width: float,
                   overwrite = False: bool
                  ): None

    + make_2d_hist(
                   fig_dict: dict,
                   figname: str,
                   axname: str,
                   savepath: str,
                   x_data: np.ndarray,
                   y_data: np.ndarray,
                   x_min: float,
                   x_max: float,
                   y_min: float,
                   y_max: float,
                   x_bin_width: float,
                   y_bin_width: float,
                   overwrite = False: bool
                  ): None

    Private methods:
    - make_bin_edges(
                     bounds: list<float>,
                     bin_width: float
                    ): np.ndarray

    - dict_print(
                 dictionary: dict,
                 num_tab = 0
                ): None

    oijijasdifjiasjdfijasdiofjaskdhasdfasdf.
    asdhfliuashlfiuahsdlifuhalsiudfha.
    asdlfhalsdfhlaskdjhflasjkdhflakdsjfh.
    '''
    
    def __init__(self, datalist_1: list | np.ndarray, datalist_2 = None):
        if not isinstance(datalist_1, (list, np.ndarray)):
            raise TypeError('datalist_1 must be a list or a np.ndarray')

        if not isinstance(datalist_2, (type(None), list, np.ndarray)):
            raise TypeError('datalist_2 must be left empty or be a list or a np.ndarray')

        if not isinstance(datalist_2, type(None)):
            if len(datalist_1) != len(datalist_2):
                raise TypeError(f'The length of datalist_1 is {len(datalist_1)} and the length of datalist_2 is {len(datalist_2)}\nThe datalists must be the same length')

        self._datalist_1 = np.array(datalist_1)
        self._datalist_2 = np.array(datalist_2)
        self._metadata = {}

    def set_datalist(self, datalist_num: int, new_datalist: list | np.ndarray):
        if datalist_num != 1 and datalist_num != 2:
            raise TypeError(f'{datalist_num} is not a valid datalist number\nThe valid datalist numbers are 1 and 2')

        if not isinstance(new_datalist, (list, np.ndarray)):
            raise TypeError(f'new_datalist for datalist_{datalist_num} must be a list or a np.ndarray')

        if datalist_num == 1:
            self._datalist_1 = np.array(new_datalist)
        if datalist_num == 2:
            self._datalist_2 = np.array(new_datalist)

    def get_datalist(self, datalist_num: int):
        if datalist_num != 1 and datalist_num != 2:
            raise TypeError(f'{datalist_num} is not a valid datalist number\nThe valid datalist numbers are 1 and 2')

        if datalist_num == 1:
            return self._datalist_1
        if datalist_num == 2:
            return self._datalist_2

    def print_datalist(self, datalist_num: int):
        if datalist_num != 1 and datalist_num != 2:
            raise TypeError(f'{datalist_num} is not a valid datalist number\nThe valid datalist numbers are 1 and 2')

        if datalist_num == 1:
            print(self._datalist_1)
        if datalist_num == 2:
            print(self._datalist_2)

    def add_field(self, fieldname: str, value):
        if type(fieldname) != str:
            raise TypeError('The fieldname must be a string')

        self._metadata[fieldname] = value

    def get_metadata(self):
        return self._metadata

    def print_metadata(self):
        self.dict_print(self._metadata)

    def fit_to_gauss_sum(self, x_data, y_data, x_scale, bounds, *init_params):

        uncertainty = np.sqrt(y_data) # assume Poissonian statistics for uncertainty on each bin count

        assert len(init_params)%3 == 0

        # sum of Gaussians fit function; each Gaussian takes 3 parameters; for N Gaussians in sum, 3N total parameters exist
        fit_func = lambda x, *params: sum([
            params[3*i] * np.exp(-1*np.square(x - params[3*i + 1])/(2*params[3*i + 2]**2))
            for i in range(int(len(params)/3))
        ])

        output = curve_fit( # calling the SciPy fit function
                           fit_func,
                           x_data,
                           y_data,
                           #sigma = uncertainty,
                           p0 = list(init_params),
                           absolute_sigma = True,
                           method = 'trf',
                           x_scale = x_scale,
                           bounds = bounds,
                           full_output = True,
                          )

        params = output[0] # list of optimized fit parameters
        param_cov = output[1] # covariance matrix of fit parameters
        info_dict = output[2] # dict of info about fit

        param_uncertainty = np.sqrt(np.diag(param_cov)) # list of uncertainties of optimized fit parameters

        residuals = info_dict['fvec'] # list of num. of sigma each data pt. is from fit; (fit - data)/data_unc

        chi_sq = sum(np.power(residuals, 2))

        return params, param_uncertainty, chi_sq

    def fit_to_parab():

        return params, param_uncertainty, chi_sq

    def plot_vert_line(self, fig_dict, figname, axname, savepath, x, ymin, ymax, linewidths = 0.75, linestyles = 'solid', label = '', legend_loc = [1,1], overwrite = False):
        if type(figname) != str or type(axname) != str:
            raise TypeError('figname and axname must both be strings')

        style = hep.style.CMS
        style['font.size'] = 13
        plt.style.use(style)

        color_list = ['#000000', '#ff0000', '#ff00cc', '#000000', '#000000',]
        #               black      red       purple       black      purple

        if overwrite == False:
            if figname in fig_dict:
                this_fig  = fig_dict[figname]['fig']
                this_ax_a = fig_dict[figname][axname]

#                if fig_dict[figname]['vert_line_ind_zero_used'] == True:
#                    fig_dict[figname]['vert_line_ind'] += 1

        if (figname not in fig_dict):
            raise TypeError('{figname} does not exist\nvertical lines cannot be plotted')
#            this_fig, this_ax_a = plt.subplots(
#                                               figsize = [7.0, 7.0],
#                                               dpi = 300,
#                                              )

#            fig_dict[figname] = {}
#            fig_dict[figname]['savepath'] = savepath
#            fig_dict[figname]['vert_line_ind'] = 0
#            fig_dict[figname]['line_plot_ind'] = 0
#            fig_dict[figname]['scatter_plot_ind'] = 0
#            fig_dict[figname]['hist_ind'] = 0
#            fig_dict[figname]['vert_line_ind_zero_used'] = False
#            fig_dict[figname]['line_plot_ind_zero_used'] = False
#            fig_dict[figname]['scatter_plot_ind_zero_used'] = False
#            fig_dict[figname]['hist_ind_zero_used'] = False

        this_ax_a.vlines(
                         x = x,
                         ymin = ymin,
                         ymax = ymax,
                         colors = color_list[fig_dict[figname]['hist_ind']],
                         linestyles = linestyles,
                         label = label,
                         linewidths = linewidths
                        )

#        fig_dict[figname]['vert_line_ind_zero_used'] = True

#        this_ax_a.set_xscale(xscale)
#        this_ax_a.set_yscale(yscale)

#        if y_bounds != None:
#            this_ax_a.set_ylim(y_bounds)
#
#        if xlabel != None:
#            this_ax_a.set_xlabel(xlabel, fontsize = 18)
#        if ylabel != None:
#            this_ax_a.set_ylabel(ylabel, fontsize = 18)
#
#        if title != None:
#            this_ax_a.set_title(title)

        '''[[r'$m_{W_{R}}$' + ' = 2000 GeV', 0.1, 0.8], ['smeared jets: ' + r'$\sigma$' + ' = 0.02', 0.1, 0.75], ['no mass width', 0.1, 0.7]] <--- format for annotations list'''
#        if annotations != None:
#            for annotation_params in annotations:
#                string = annotation_params[0]
#                x_frac = annotation_params[1]
#                y_frac = annotation_params[2]

#                this_ax_a.annotate(string, (x_frac, y_frac), xycoords = 'axes fraction', fontsize = 16)

        if len(label) > 0:
            this_ax_a.legend(
                             loc = 'upper right',
                             bbox_to_anchor = (legend_loc[0], legend_loc[1]),
                             bbox_transform = this_ax_a.transAxes,
                             fontsize = 16,
                            )

        this_fig.tight_layout()

#        fig_dict[figname]['fig'] = this_fig
#        fig_dict[figname][axname] = this_ax_a

    def make_line_plot(self, fig_dict, figname, axname, savepath, label, legend_loc, y_data, x_data = None, x_bounds = None, y_bounds = None, xscale = 'linear', yscale = 'linear', xlabel = None, ylabel = None, title = None, annotations = None, overwrite = False): # fig_dict comes from the FigAgg object
        if type(figname) != str or type(axname) != str:
            raise TypeError('figname and axname must both be strings')
        if type(y_data) != np.ndarray:
            raise TypeError('y_data must be a numpy.ndarray')
        if not isinstance(x_data, (type(None), np.ndarray)):
            raise TypeError('x_data must either be left blank or must be a numpy.ndarray')
        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean')

        color_list = ['#000000', '#FE019A', '#400321',]
        #              black      pink       purple

        if overwrite == False:
            if figname in fig_dict:
                this_fig  = fig_dict[figname]['fig']
                this_ax_a = fig_dict[figname][axname]

                if fig_dict[figname]['line_plot_ind_zero_used'] == True:
                    fig_dict[figname]['line_plot_ind'] += 1

        if (figname not in fig_dict or overwrite == True):
            this_fig, this_ax_a = plt.subplots(
                                               figsize = [7.0, 7.0],
                                               dpi = 100,
                                              )

            fig_dict[figname] = {}
            fig_dict[figname]['savepath'] = savepath
            fig_dict[figname]['line_plot_ind'] = 0
            fig_dict[figname]['scatter_plot_ind'] = 0
            fig_dict[figname]['hist_ind'] = 0
            fig_dict[figname]['line_plot_ind_zero_used'] = False
            fig_dict[figname]['scatter_plot_ind_zero_used'] = False
            fig_dict[figname]['hist_ind_zero_used'] = False

        if isinstance(x_data, type(None)):
            n = len(y_data)
            x_data = np.linspace(1, n, n)

        style = hep.style.CMS
        style['font.size'] = 13
        plt.style.use(style)

        this_ax_a.plot(
                       x_data,
                       y_data,
                       linewidth = 1,
                       color = color_list[fig_dict[figname]['line_plot_ind']],
                       label = label,
                      )
        fig_dict[figname]['line_plot_ind_zero_used'] = True

        if x_bounds != None:
            this_ax_a.set_xlim(x_bounds)
        if y_bounds != None:
            this_ax_a.set_ylim(y_bounds)

        this_ax_a.set_xscale(xscale)
        this_ax_a.set_yscale(yscale)

        if xlabel != None:
            this_ax_a.set_xlabel(xlabel, fontsize = 18)
        if ylabel != None:
            this_ax_a.set_ylabel(ylabel, fontsize = 18)

        if title != None:
            this_ax_a.set_title(title)

        '''[[r'$m_{W_{R}}$' + ' = 2000 GeV', 0.1, 0.8], ['smeared jets: ' + r'$\sigma$' + ' = 0.02', 0.1, 0.75], ['no mass width', 0.1, 0.7]] <--- format for annotations list'''
        if annotations != None:
            for annotation_params in annotations:
                string = annotation_params[0]
                x_frac = annotation_params[1]
                y_frac = annotation_params[2]

                this_ax_a.annotate(
                                   string,
                                   (x_frac, y_frac),
                                   xycoords = 'axes fraction',
                                   fontsize = 16
                                  )

        this_ax_a.legend(
                         loc = 'upper right',
                         bbox_to_anchor = (legend_loc[0], legend_loc[1]),
                         bbox_transform = this_ax_a.transAxes,
                         fontsize = 16,
                        )
        this_fig.tight_layout()

        fig_dict[figname]['fig'] = this_fig
        fig_dict[figname][axname] = this_ax_a

    def make_scatterplot(self, fig_dict, figname, axname, savepath, label, legend_loc, y_data, x_data = None, x_bounds = None, y_bounds = None, xscale = 'linear', yscale = 'linear', xlabel = None, ylabel = None, title = None, annotations = None, overwrite = False): # fig_dict comes from the FigAgg object
        if type(figname) != str or type(axname) != str:
            raise TypeError('figname and axname must both be strings')
        if type(y_data) != np.ndarray:
            raise TypeError('y_data must be a numpy.ndarray')
        if not isinstance(x_data, (type(None), np.ndarray)):
            raise TypeError('x_data must either be left blank or must be a numpy.ndarray')
        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean')

        color_list = ['#000000', '#FE019A', '#400321',]
        #              black      pink       purple

        if overwrite == False:
            if figname in fig_dict:
                this_fig  = fig_dict[figname]['fig']
                this_ax_a = fig_dict[figname][axname]

                if fig_dict[figname]['scatter_plot_ind_zero_used'] == True:
                    fig_dict[figname]['scatter_plot_ind'] += 1

        if (figname not in fig_dict or overwrite == True):
            this_fig, this_ax_a = plt.subplots(
                                               figsize = [7.0, 7.0],
                                               dpi = 100,
                                              )

            fig_dict[figname] = {}
            fig_dict[figname]['savepath'] = savepath
            fig_dict[figname]['line_plot_ind'] = 0
            fig_dict[figname]['scatter_plot_ind'] = 0
            fig_dict[figname]['hist_ind'] = 0
            fig_dict[figname]['line_plot_ind_zero_used'] = False
            fig_dict[figname]['scatter_plot_ind_zero_used'] = False
            fig_dict[figname]['hist_ind_zero_used'] = False

        if isinstance(x_data, type(None)):
            n = len(y_data)
            x_data = np.linspace(1, n, n)

        style = hep.style.CMS
        style['font.size'] = 13
        plt.style.use(style)

        this_ax_a.scatter(
                          x_data,
                          y_data,
                          s = 1,
                          color = color_list[fig_dict[figname]['line_plot_ind']],
                          label = label,
                         )
        fig_dict[figname]['scatter_plot_ind_zero_used'] = True

        if x_bounds != None:
            this_ax_a.set_xlim(x_bounds)
        if y_bounds != None:
            this_ax_a.set_ylim(y_bounds)

        this_ax_a.set_xscale(xscale)
        this_ax_a.set_yscale(yscale)

        if xlabel != None:
            this_ax_a.set_xlabel(xlabel, fontsize = 18)
        if ylabel != None:
            this_ax_a.set_ylabel(ylabel, fontsize = 18)

        if title != None:
            this_ax_a.set_title(title)

        '''[[r'$m_{W_{R}}$' + ' = 2000 GeV', 0.1, 0.8], ['smeared jets: ' + r'$\sigma$' + ' = 0.02', 0.1, 0.75], ['no mass width', 0.1, 0.7]] <--- format for annotations list'''
        if annotations != None:
            for annotation_params in annotations:
                string = annotation_params[0]
                x_frac = annotation_params[1]
                y_frac = annotation_params[2]

                this_ax_a.annotate(
                                   string,
                                   (x_frac, y_frac),
                                   xycoords = 'axes fraction',
                                   fontsize = 16
                                  )

        this_ax_a.legend(
                         loc = 'upper right',
                         bbox_to_anchor = (legend_loc[0], legend_loc[1]),
                         bbox_transform = this_ax_a.transAxes,
                         fontsize = 16,
                        )
        this_fig.tight_layout()

        fig_dict[figname]['fig'] = this_fig
        fig_dict[figname][axname] = this_ax_a

    def make_1d_hist(self, fig_dict, figname, axname, savepath, label, legend_loc, data, bin_width, x_bounds, y_bounds = None, xscale = 'linear', yscale = 'linear', xlabel = None, ylabel = None, title = None, annotations = None, overwrite = False):
        if type(figname) != str or type(axname) != str:
            raise TypeError('figname and axname must both be strings')
        if type(data) != np.ndarray:
            raise TypeError('data must be a numpy.ndarray')
        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean')

        style = hep.style.CMS
        style['font.size'] = 13
        plt.style.use(style)

        color_list = ['#1eb935', '#1e25b9', '#b96c1e', '#000000', '#7c1eb9',]
        #               green      blue       rust       black      purple

        if overwrite == False:
            if figname in fig_dict:
                this_fig  = fig_dict[figname]['fig']
                this_ax_a = fig_dict[figname][axname]

                if fig_dict[figname]['hist_ind_zero_used'] == True:
                    fig_dict[figname]['hist_ind'] += 1

        if (figname not in fig_dict or overwrite == True):
            this_fig, this_ax_a = plt.subplots(
                                               figsize = [7.0, 7.0],
                                               dpi = 100,
                                              )

            fig_dict[figname] = {}
            fig_dict[figname]['savepath'] = savepath
            fig_dict[figname]['line_plot_ind'] = 0
            fig_dict[figname]['scatter_plot_ind'] = 0
            fig_dict[figname]['hist_ind'] = 0
            fig_dict[figname]['line_plot_ind_zero_used'] = False
            fig_dict[figname]['scatter_plot_ind_zero_used'] = False
            fig_dict[figname]['hist_ind_zero_used'] = False

        bin_edges = self.make_bin_edges(
                                        bounds = x_bounds,
                                        bin_width = bin_width
                                       )

        bin_data = this_ax_a.hist(
                                               data,
                                               color = color_list[fig_dict[figname]['hist_ind']],
                                               bins = bin_edges,
                                               histtype = 'step',
                                               label = label
                                              )
        fig_dict[figname]['hist_ind_zero_used'] = True

        bin_counts = bin_data[0]
        bin_edges  = bin_data[1]

        this_ax_a.set_xscale(xscale)
        this_ax_a.set_yscale(yscale)

        if y_bounds != None:
            this_ax_a.set_ylim(y_bounds)

        if xlabel != None:
            this_ax_a.set_xlabel(xlabel, fontsize = 18)
        if ylabel != None:
            this_ax_a.set_ylabel(ylabel, fontsize = 18)

        if title != None:
            this_ax_a.set_title(title)

        '''[[r'$m_{W_{R}}$' + ' = 2000 GeV', 0.1, 0.8], ['smeared jets: ' + r'$\sigma$' + ' = 0.02', 0.1, 0.75], ['no mass width', 0.1, 0.7]] <--- format for annotations list'''
        if annotations != None:
            for annotation_params in annotations:
                string = annotation_params[0]
                x_frac = annotation_params[1]
                y_frac = annotation_params[2]

                this_ax_a.annotate(string, (x_frac, y_frac), xycoords = 'axes fraction', fontsize = 16)

        this_ax_a.legend(
                         loc = 'upper right',
                         bbox_to_anchor = (legend_loc[0], legend_loc[1]),
                         bbox_transform = this_ax_a.transAxes,
                         fontsize = 16,
                        )
        this_fig.tight_layout()

        fig_dict[figname]['fig'] = this_fig
        fig_dict[figname][axname] = this_ax_a

        return bin_counts, bin_edges

    def make_2d_hist(self, fig_dict, figname, axname, savepath, x_data, y_data, x_min, x_max, y_min, y_max, x_bin_width, y_bin_width, xscale = 'linear', yscale = 'linear', xlabel = None, ylabel = None, overwrite = False):
        if type(figname) != str or type(axname) != str:
            raise TypeError('figname and axname must both be strings')
        if type(x_data) != np.ndarray:
            raise TypeError('x_data must be a numpy.ndarray')
        if type(y_data) != np.ndarray:
            raise TypeError('y_data must be a numpy.ndarray')
        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean')

        if overwrite == False:
            if figname in fig_dict:
                raise TypeError(f'{figname} already exists in the figure aggregate\nSet overwrite to True to overwrite the current figure')

        style = hep.style.CMS
        style['font.size'] = 13
        plt.style.use(style)

        this_fig, this_ax_a = plt.subplots(
                                           figsize = [6.0, 6.0],
                                           dpi = 100,
                                          )

        x_bin_edges = self.make_bin_edges(
                                          bounds = [x_min, x_max],
                                          bin_width = x_bin_width
                                         )

        y_bin_edges = self.make_bin_edges(
                                          bounds = [y_min, y_max],
                                          bin_width = y_bin_width
                                         )

        this_ax_a.hist2d(
                         x_data,
                         y_data,
                         bins = [x_bin_edges, y_bin_edges],
                         cmap = 'viridis',
                        )

        this_ax_a.set_xscale(xscale)
        this_ax_a.set_yscale(yscale)

        if xlabel != None:
            this_ax_a.set_xlabel(xlabel, fontsize = 18)
        if ylabel != None:
            this_ax_a.set_ylabel(ylabel, fontsize = 18)

        fig_dict[figname] = {}
        fig_dict[figname]['fig'] = this_fig
        fig_dict[figname][axname] = this_ax_a
        fig_dict[figname]['savepath'] = savepath

    def make_bin_edges(self, bounds, bin_width):

        curr_edge = bounds[0]

        bin_edges = [curr_edge]

        while max(bin_edges) < bounds[1]:
            curr_edge += bin_width

            bin_edges.append(curr_edge)

        bin_edges = np.array(bin_edges)

        return bin_edges

    def dict_print(self, dictionary: dict, num_tab = 0):
        if type(dictionary) != dict:
            raise TypeError('Argument to dict_print must be a dict')

        num_tab += 1

        tab = '    '

        for key, value in dictionary.items():
            print(tab*(num_tab-1) + f'{key}:')
            if str(type(value)) == "<class 'dict'>":
                self.dict_print(value, num_tab)
            else:
                print(tab*num_tab + f'{value}\n')

