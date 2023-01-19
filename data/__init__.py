import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.cityscapes_loader_single import cityscapesLoader_single
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "cityscapes_single": cityscapesLoader_single,
        "gta": GTA5DataSet,
        "syn": SynthiaDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return './data/city/'
    if name == 'cityscapes_single':
        return './data/city/'
    if name == 'gta' or name == 'gtaUniform':
        return './data/gtav/'
    if name == 'syn':
        return './data/syn/'
