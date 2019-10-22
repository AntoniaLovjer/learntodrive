import numpy as np
import torch

import pandas as pd

from learntodrive.utils import move_target_to_cuda
from learntodrive.utils import move_data_to_cuda
from learntodrive.utils import get_features_3

def add_results(results, predictionSteer, predictionSpeed, front_name, config, normalie_steer=True):
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']
    steering = np.squeeze(predictionSteer['canSteering'].cpu().data.numpy())
    speed = np.squeeze(predictionSpeed['canSpeed'].cpu().data.numpy())
    last_image_names = front_name[0]
    front_name = [x for x in last_image_names]
    image_front_name = np.squeeze(np.array(front_name))
    # print(image_front_name)
    if normalize_targets:
        if normalie_steer:
            steering = (steering*target_std['canSteering'])+target_mean['canSteering']
        speed = (speed*target_std['canSpeed'])+target_mean['canSpeed']
        front_name = front_name
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)
    results['cameraFront'].extend(front_name)

def predict_submission(modelSteer, modelSpeed, pickle_files, test_loader, config, train_loader=None):
    results = {'canSteering': [],
           'canSpeed': [], 
           'cameraFront': []}
    
    modelSteer.eval()
    modelSpeed.eval()
    with torch.no_grad():
        for batch_idx, (data, target, front_name) in enumerate(test_loader):
            data = move_data_to_cuda(data)
            target = move_target_to_cuda(target)
            hidden_features = [get_features_3(front_name, x).cuda() for x in pickle_files]
            predictionSteer = modelSteer(data, hidden_features)
            predictionSpeed = modelSpeed(data, hidden_features)
            if train_loader!=None:
                a = np.array(torch.argmax(predictionSteer['canSteering'], axis=1).detach().cpu())
                means_steer = np.vectorize(train_loader.drive360.idx_to_mean.__getitem__)(a)
                predictionSteer['canSteering'] = torch.from_numpy(means_steer)
            add_results(results, predictionSteer, predictionSpeed, front_name, config, train_loader==None)
    df = pd.DataFrame.from_dict(results)
    return(df)

def create_submission(df, filename_input, filename_output):
    """
        The data is sampled with a ration of 1:10. To avoid data leackage, we extrapolate the values. The first 20 frames are filled with the average.
        
        The extrapolation shifts the predicted values (speed and angle) and index (img number) by 2 in the sampled dataset. After merging the sampled values with the original dataframe, these shifted values are "back filled". These backed filled values are not data leackage, as they were predicted only past values. Afterwards the extrapolation is done with a linear approach.
    """
    test_full = pd.read_csv(filename_input)
    df['img_idx'] = df['cameraFront'].str[-9:].str[:5].astype(int)
    df['canSpeed_1'] = df['canSpeed'].shift(1)
    df['canSpeed_2'] = df['canSpeed'].shift(2)
    df['canSteering_1'] = df['canSteering'].shift(1)
    df['canSteering_2'] = df['canSteering'].shift(2)
    df['img_idx_1'] = df['img_idx'].shift(1)
    df['img_idx_2'] = df['img_idx'].shift(2)    
    complete_test_set = pd.merge(test_full, df, how='left', on='cameraFront')
    index_list = complete_test_set.groupby('chapter').apply(lambda x: x.iloc[100:]).index.droplevel(level=0).tolist()
    complete_test_set = complete_test_set.loc[index_list,]

    complete_test_set = pd.merge(test_full, df, how='left', on='cameraFront')
    complete_test_set['img_idx'] = complete_test_set['cameraFront'].str[-9:].str[:5].astype(int)
    index_list = complete_test_set.groupby('chapter').apply(lambda x: x.iloc[100:]).index.droplevel(level=0).tolist()
    complete_test_set = complete_test_set.loc[index_list,]
    complete_test_set['flag'] = complete_test_set['flag'].fillna(0)
    
    complete_test_set['canSpeed_1'] = complete_test_set.groupby(['chapter'])['canSpeed_1'].bfill(limit=10)
    complete_test_set['canSpeed_2'] = complete_test_set.groupby(['chapter'])['canSpeed_2'].bfill(limit=10)
    complete_test_set['canSteering_1'] = complete_test_set.groupby(['chapter'])['canSteering_1'].bfill(limit=10)
    complete_test_set['canSteering_2'] = complete_test_set.groupby(['chapter'])['canSteering_2'].bfill(limit=10)
    complete_test_set['img_idx_1'] = complete_test_set.groupby(['chapter'])['img_idx_1'].bfill(limit=10)
    complete_test_set['img_idx_2'] = complete_test_set.groupby(['chapter'])['img_idx_2'].bfill(limit=10)
    
    complete_test_set['extrapolatedStreering'] = (complete_test_set['canSteering_2'] complete_test_set['canSteering_1'])/(complete_test_set['img_idx_2']-complete_test_set['img_idx_1']) * (complete_test_set['img_idx']-complete_test_set['img_idx_1']) + complete_test_set['canSteering_1']
    complete_test_set['extrapolatedSpeed'] = (complete_test_set['canSpeed_2']-complete_test_set['canSpeed_1'])/(complete_test_set['img_idx_2']-complete_test_set['img_idx_1']) * (complete_test_set['img_idx']-complete_test_set['img_idx_1']) + complete_test_set['canSpeed_1']
    
    complete_test_set.loc[complete_test_set['canSpeed'].isna(),'canSpeed'] = complete_test_set.loc[complete_test_set['canSpeed'].isna(),'extrapolatedSpeed']
    complete_test_set.loc[complete_test_set['canSteering'].isna(),'canSteering'] = complete_test_set.loc[complete_test_set['canSteering'].isna(),'extrapolatedStreering']
    
    complete_test_set['canSpeed'] = complete_test_set['canSpeed'].fillna(13.426163367846936)
    complete_test_set['canSteering'] = complete_test_set['canSteering'].fillna(-5.406788214535221)
    
    if np.sum(np.sum(complete_test_set.isna())>0)>0:
        print('Error some NA values!')
        complete_test_set.to_csv('Submissions/error_submission.csv', index=False)
    else:
        if complete_test_set.shape[0] != 279863:
            print('Sumbission file has wrong number of lines')
        else:
            print('Submission file has no NAs!')
            # pick out only the last two columns
            submission_cols = complete_test_set[['canSteering', 'canSpeed']]
            # to cvs final submission
            submission_cols.to_csv(filename_output, index=False)

def create_validation(df, filename_input, filename_output):
    """
        The data is sampled with a ration of 1:10. To avoid data leackage, we extrapolate the values. The first 20 frames are filled with the average.
        
        The extrapolation shifts the predicted values (speed and angle) and index (img number) by 2 in the sampled dataset. After merging the sampled values with the original dataframe, these shifted values are "back filled". These backed filled values are not data leackage, as they were predicted only past values. Afterwards the extrapolation is done with a linear approach.
    """
    df['flag'] = 1
    test_full = pd.read_csv(filename_input)
    df['img_idx'] = df['cameraFront'].str[-9:].str[:5].astype(int)
    df['canSpeed_1'] = df['canSpeed'].shift(1)
    df['canSpeed_2'] = df['canSpeed'].shift(2)
    df['canSteering_1'] = df['canSteering'].shift(1)
    df['canSteering_2'] = df['canSteering'].shift(2)
    df['img_idx_1'] = df['img_idx'].shift(1)
    df['img_idx_2'] = df['img_idx'].shift(2)    
    if 'canSteering' in list(test_full.columns):
        test_full['canSteering_correct'] = test_full['canSteering']
        test_full = test_full.drop(['canSteering'], axis=1)
    if 'canSpeed' in list(test_full.columns):
        test_full['canSpeed_correct'] = test_full['canSpeed']
        test_full = test_full.drop(['canSpeed'], axis=1)
    complete_test_set = pd.merge(test_full, df, how='left', on='cameraFront')
    complete_test_set['img_idx'] = complete_test_set['cameraFront'].str[-9:].str[:5].astype(int)
    index_list = complete_test_set.groupby('chapter').apply(lambda x: x.iloc[100:]).index.droplevel(level=0).tolist()
    complete_test_set = complete_test_set.loc[index_list,]
    complete_test_set['flag'] = complete_test_set['flag'].fillna(0)
    
    complete_test_set['canSpeed_1'] = complete_test_set.groupby(['chapter'])['canSpeed_1'].bfill(limit=10)
    complete_test_set['canSpeed_2'] = complete_test_set.groupby(['chapter'])['canSpeed_2'].bfill(limit=10)
    complete_test_set['canSteering_1'] = complete_test_set.groupby(['chapter'])['canSteering_1'].bfill(limit=10)
    complete_test_set['canSteering_2'] = complete_test_set.groupby(['chapter'])['canSteering_2'].bfill(limit=10)
    complete_test_set['img_idx_1'] = complete_test_set.groupby(['chapter'])['img_idx_1'].bfill(limit=10)
    complete_test_set['img_idx_2'] = complete_test_set.groupby(['chapter'])['img_idx_2'].bfill(limit=10)
    
    complete_test_set['extrapolatedStreering'] = (complete_test_set['canSteering_2'] complete_test_set['canSteering_1'])/(complete_test_set['img_idx_2']-complete_test_set['img_idx_1']) * (complete_test_set['img_idx']-complete_test_set['img_idx_1']) + complete_test_set['canSteering_1']
    complete_test_set['extrapolatedSpeed'] = (complete_test_set['canSpeed_2']-complete_test_set['canSpeed_1'])/(complete_test_set['img_idx_2']-complete_test_set['img_idx_1']) * (complete_test_set['img_idx']-complete_test_set['img_idx_1']) + complete_test_set['canSpeed_1']
    
    complete_test_set.loc[complete_test_set['canSpeed'].isna(),'canSpeed'] = complete_test_set.loc[complete_test_set['canSpeed'].isna(),'extrapolatedSpeed']
    complete_test_set.loc[complete_test_set['canSteering'].isna(),'canSteering'] = complete_test_set.loc[complete_test_set['canSteering'].isna(),'extrapolatedStreering']
    
    complete_test_set['canSpeed'] = complete_test_set['canSpeed'].fillna(13.426163367846936)
    complete_test_set['canSteering'] = complete_test_set['canSteering'].fillna(-5.406788214535221)
    
    complete_test_set.to_csv(filename_output, index=False)