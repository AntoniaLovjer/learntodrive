import numpy as np
import torch

from learntodrive.utils import move_target_to_cuda
from learntodrive.utils import move_data_to_cuda
from learntodrive.utils import get_features_3

def run_validation(model, validation_loader, config, pickle_files, train_loader=None):
  target_mean = config['target']['mean']
  target_std = config['target']['std']
  model.eval()
  target_speed = np.array([])
  target_steer = np.array([])
  pred_speed = np.array([])
  pred_steer = np.array([])
  with torch.no_grad():
      for batch_idx, (data, target, front_name) in enumerate(validation_loader):
          data = move_data_to_cuda(data)
          target = move_target_to_cuda(target)
          hidden_features = [get_features_3(front_name, x).cuda() for x in pickle_files]
          prediction = model(data, hidden_features)
          if train_loader!=None:
              a = np.array(torch.argmax(prediction['canSteering'], axis=1).detach().cpu())
              means_steer = np.vectorize(train_loader.drive360.idx_to_mean.__getitem__)(a)
              cur_pred_steer = means_steer
          else:
              cur_pred_steer = np.asarray(prediction['canSteering'].detach().cpu())
          cur_pred_speed = np.asarray(prediction['canSpeed'].detach().cpu())
          cur_target_speed = np.asarray(target['canSpeed'].detach().cpu())
          cur_target_steer = np.asarray(target['canSteering'].detach().cpu())
          if train_loader==None:
              cur_pred_steer = (cur_pred_steer*target_std['canSteering'])+target_mean['canSteering']
          cur_target_steer = (cur_target_steer*target_std['canSteering'])+target_mean['canSteering']
          cur_pred_speed = (cur_pred_speed*target_std['canSpeed'])+target_mean['canSpeed']
          cur_target_speed = (cur_target_speed*target_std['canSpeed'])+target_mean['canSpeed']
          pred_speed = np.concatenate([pred_speed, cur_pred_speed])
          ####### Fixed #######
          pred_steer = np.concatenate([pred_steer, cur_pred_steer])
          target_speed = np.concatenate([target_speed, cur_target_speed])
          target_steer = np.concatenate([target_steer, cur_target_steer])

          if (batch_idx+1) % 10 == 0:
                  print("Validation batch: " + str(batch_idx+1))
  mse_steer = (np.square(pred_steer - target_steer)).mean()
  mse_speed = (np.square(pred_speed - target_speed)).mean()
  return(mse_steer, mse_speed)