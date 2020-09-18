"""
python module I (bill bai) created for convenience when creating and training
models.
"""
import os
import torch
import argparse


'''OS RELATED UTILS'''
def create_dir(dir_name):
    """
    Create directory if it doensn't already exist
    :param dir_name: directory name
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def str2bool(val):
    """
    Returns a boolean value based on a string passed through command line
    :param val: boolean string
    :return: True or False
    """
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_mode(train, val):
    if (train and val) or (not train and not val):
        raise Exception('Please enter either train or eval mode.')
    elif train:
        return 'train'
    elif val:
        return 'eval'


'''Logging'''
def log_data_to_txt(filename, msg):
    """
    Log data (eg training data)
    :param filename: log file
    :param msg: message to log
    """
    with open(f'{filename}.txt', 'a') as logfile:
        print(msg)
        logfile.write('\n' + "-" * 20)
        logfile.write(f'{msg}')


'''PYTORCH UTILS'''
def filter_gradients(model):
    """
    Filters gradients, only passes params that require grad
    :param model: the model
    :return: filtered gradients
    """
    return filter(
        lambda param: param.requires_grad
        , model.parameters()
    )

def save_state_dict(model, name, dir='./pt'):
    """
    saves the statedict of a model
    :param model: model or optimizer to save
    :param name: name to save under
    :param dir: directory to save in (defaulted to './pt/')
    """
    create_dir(dir)
    torch.save(model.state_dict(), f'{dir}/{name}.pt')
