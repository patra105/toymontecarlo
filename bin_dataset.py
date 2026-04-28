'''This class is for histogram bin data (bin edges and bin counts) manipulation.'''
import numpy as np
from scipy.optimize import curve_fit

class BinDataSet:
    '''
    (c) Sean Poczos 2025
    '''
    
    def __init__(self, bin_edges: list | np.ndarray, bin_counts: list | np.ndarray):
        if not isinstance(bin_edges, (list, np.ndarray)):
            raise TypeError('bin_edges must be a list or a np.ndarray')

        if not isinstance(bin_counts, (list, np.ndarray)):
            raise TypeError('bin_counts must be a list or a np.ndarray')

        if len(bin_edges) != (len(bin_counts) + 1):
            raise TypeError(f'The length of bin_edges is {len(bin_edges)} and the length of bin_counts is {len(bin_counts)}\nbin_edges must be 1 longer than bin_counts')

        self._bin_edges = np.array(bin_edges)
        self._bin_counts = np.array(bin_counts)
        self._metadata = {}

    def set_datalist(self, datalist_num: int, new_datalist: list | np.ndarray):
        if datalist_num != 1 and datalist_num != 2:
            raise TypeError(f'{datalist_num} is not a valid datalist number\nThe valid datalist numbers are 1 and 2')

        if not isinstance(new_datalist, (list, np.ndarray)):
            raise TypeError(f'new_datalist for datalist_{datalist_num} must be a list or a np.ndarray')

        if datalist_num == 1:
            self._bin_edges = np.array(new_datalist)
        if datalist_num == 2:
            self._bin_counts = np.array(new_datalist)

    def get_datalist(self, datalist_num: int):
        if datalist_num != 1 and datalist_num != 2:
            raise TypeError(f'{datalist_num} is not a valid datalist number\nThe valid datalist numbers are 1 and 2')

        if datalist_num == 1:
            return self._bin_edges
        if datalist_num == 2:
            return self._bin_counts

    def print_bin_edges(self):
        print(self._bin_edges)

    def print_bin_counts(self):
        print(self._bin_counts)

    def add_field(self, fieldname: str, value):
        if type(fieldname) != str:
            raise TypeError('The fieldname must be a string')

        self._metadata[fieldname] = value

    def get_metadata(self):
        return self._metadata

    def print_metadata(self):
        self.dict_print(self._metadata)

    def get_bin_centers(self):
        
        bin_centers = 0.5*(self._bin_edges[0:-1] + self._bin_edges[1:])

        return bin_centers

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

    # returns the indices of the first bins with a count below a certain value both to the left and right of some prechosen start bin
    def find_first_bin_below_ind(self, start_ind, min_count, start_at_max = True, min_count_type = 'raw'):
        if start_at_max == True:
            start_ind = np.argmax(self._bin_counts)

        if min_count_type == 'raw':
            min_val = min_count
        if min_count_type == 'frac':
            min_val = min_count*self._bin_counts[start_ind]

        if self._bin_counts[start_ind] <= min_val:
            raise TypeError(f'The count at bin No. {start_ind} is already <= {min_val}')

        left_bin_counts = self._bin_counts[0:start_ind] # the bin counts to the left of the start bin
        right_bin_counts = self._bin_counts[start_ind + 1:] # the bin counts to the right of the start bin

        left_inds = np.where(left_bin_counts <= min_val)[0]
        right_inds = np.where(right_bin_counts <= min_val)[0] + start_ind + 1

        if len(left_inds) != 0:
            first_left_ind = left_inds[-1]
        else:
            first_left_ind = -1

        if len(right_inds) != 0:
            first_right_ind = right_inds[0]
        else:
            first_right_ind = start_ind

        return first_left_ind, first_right_ind

