"""
Datasets module
"""
__all__ = ['anes96', 'cancer', 'ccard', 'china_smoking', 'co2', 'committee',
           'copper', 'cpunish', 'elnino', 'engel', 'fair', 'fertility',
           'grunfeld', 'heart', 'interest_inflation', 'longley',
           'macrodata', 'modechoice', 'nile', 'randhie', 'scotland', 'spector',
           'stackloss', 'star98', 'strikes', 'sunspots', 'statecrime',
           'get_rdataset', 'get_data_home', 'clear_data_home',
           'webuse', 'check_internet']
from . import (anes96, cancer, committee, ccard, copper, cpunish, elnino,
               engel, grunfeld, interest_inflation, longley, macrodata,
               modechoice, nile, randhie, scotland, spector, stackloss,
               star98, strikes, sunspots, fair, heart, statecrime, co2,
               fertility, china_smoking)
from .utils import (get_rdataset, get_data_home, clear_data_home,
                    webuse, check_internet)
