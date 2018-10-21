from os import path


DIRPATH = path.join(path.dirname(__file__), '..', 'data/')


def find_datafiles(file_name):
    datapath = path.join(path.abspath(DIRPATH), file_name)
    return path.abspath(datapath)
