import pickle
import torch

import numpy as np

def loadPickle(file):
    with open(file, 'rb') as handle:
        return(pickle.load(handle))

def move_target_to_cuda(target):
    output = {}
    output['canSpeed'] = target['canSpeed'].cuda()
    output['canSteering'] = target['canSteering'].cuda()
    return(output)


def move_data_to_cuda(data):
    output = {}
    for dkeys1 in list(data.keys()):
        if not(dkeys1 in output):
            output[dkeys1] = {}
        for dkeys2 in list(data[dkeys1].keys()):
            output[dkeys1][dkeys2] = data[dkeys1][dkeys2].cuda()
    return(output)

def log_textfile(filename, text):
    """
    Function log_to_textfile
    
    Appends a text to a file (logs)
    
    Args:
        filename (str): Filename of logfile
        text (str): New information to log (append)
    
    Return:
    
    """
    print(text)
    f = open(filename, "a")
    f.write(str(text) + str('\n'))
    f.close()

def get_features_3(list_names, feature_dict):
    feature_output = []
    for x in list_names:
        tmp_output = []
        for y in x:
            tmp_output.append(y)
        feature_output.append(tmp_output)
    n_dim = feature_dict[list(feature_dict.keys())[0]].shape[0]
    flatten_bl = len(feature_dict[list(feature_dict.keys())[0]].shape)
    feature_output = np.asarray(feature_output)
    feature_output_2 = np.zeros(feature_output.shape + (n_dim,))
    counter = 0
    counter_2 = 0
    for x in list_names:
        for y in x:
            tmp_tmp = feature_dict[feature_output[counter, counter_2]]
            if flatten_bl>1:
                tmp_tmp = np.squeeze(tmp_tmp)
            feature_output_2[counter, counter_2] = tmp_tmp
            counter_2 += 1
        counter += 1
        counter_2 = 0
    feature_output_2 = torch.from_numpy(feature_output_2).float()
    return(feature_output_2)