'''This class is an aggregate of matplotlib.pyplot Figure and Axes objects. Its purpose is to store the objects and save the figures.'''
import matplotlib.pyplot as plt
import mplhep as hep
import re, os

class FigAgg:
    '''
    (c) Sean Poczos 2025

    Attributes:
    - _fig_dict: dict

    Public methods:
    + get_fig_dict(): dict
    + get_fig(fig_name: str): Figure
    + get_ax(fig_name: str, ax_name: str): Axes
    + print_fig_dict(): None
    + save_all_figs(): None

    Private methods:
    - dict_print(dictionary: dict, num_tab: int): None

    oijijasdifjiasjdfijasdiofjaskdhasdfasdf.
    asdhfliuashlfiuahsdlifuhalsiudfha.
    asdlfhalsdfhlaskdjhflasjkdhflakdsjfh.
    '''

    def __init__(self):
        self._fig_dict = {}

    def get_fig_dict(self):
        return self._fig_dict

    def get_fig(self, fig_name: str):
        if not isinstance(fig_name, str):
            raise TypeError('fig_name must be a str')

        if fig_name not in self._fig_dict:
            raise TypeError(f'{fig_name} is not in the figure aggregate')
        else:
            return self._fig_dict[fig_name]['fig']

    def get_ax(self, fig_name: str, ax_name: str):
        if not isinstance(fig_name, str):
            raise TypeError('fig_name must be a str')
        if not isinstance(ax_name, str):
            raise TypeError('ax_name must be a str')

        if fig_name not in self._fig_dict:
            raise TypeError(f'{fig_name} is not in the figure aggregate')
        else:
            if ax_name not in self._fig_dict[fig_name]:
                raise TypeError(f'{ax_name} is not in {fig_name}')
            else:
                return self._fig_dict[fig_name][ax_name]

    def print_fig_dict(self):
        self.dict_print(self._fig_dict)

    def save_all_figs(self):
        for fig_name, obj_dict in self._fig_dict.items():
            savepath = obj_dict['savepath']
            filename_match_obj = re.search(r"[\w]+\.[\w]+", savepath)
            filename_start_ind = filename_match_obj.start()

            direc = savepath[0:filename_start_ind]
            os.makedirs(direc, exist_ok = True)
            obj_dict['fig'].savefig(savepath)

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

