from torchvision import models
import torch.nn as nn
import torch

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class BiGRU(nn.Module):
    def __init__(self, num_cnn_features=512, num_lstm_layers=3, hidden_lstm_size=64, num_classes=3, p_dropout=0.1):
        super(BiGRU, self).__init__()
        self.num_classes = num_classes
        final_concat_size = 0
        # Main CNN
        # cnn = models.resnet34(pretrained=True)
        # cnn = ResNet34()
        # set_parameter_requires_grad(cnn, True)
        # self.features = nn.Sequential(*list(cnn.children())[:-1])
        
        # self.features = cnn
        self.intermediate = nn.Sequential(
            nn.Linear(num_cnn_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        final_concat_size += 128
        
        self.intermediate_2 = nn.Sequential(
            nn.Linear(num_cnn_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())

        # Forward LSTM
        self.lstm_forward = nn.GRU(input_size=128,
                            hidden_size=hidden_lstm_size,
                            num_layers=num_lstm_layers,
                            batch_first=False)
        final_concat_size += hidden_lstm_size
        
        # Backward LSTM
        self.lstm_backward = nn.GRU(input_size=128,
                            hidden_size=hidden_lstm_size,
                            num_layers=num_lstm_layers,
                            batch_first=False)
        final_concat_size += hidden_lstm_size
        
        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, data, hidden_features=None):
        module_outputs = []
        lstm_f = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        for k, v in data['cameraFront'].items():
            tmp_output = []
            for i in range(len(hidden_features)):
                tmp_output.append(hidden_features[i][k,])
            x_2 = torch.cat(tmp_output, 1)
            x = self.intermediate(x_2)
            lstm_f.append(x)
            if k == 0:
                module_outputs.append(self.intermediate_2(x_2))
                
        # reverse the list of front images
        lstm_b = lstm_f[::-1]
        # feed the reversed images into the lstm layer
        b_lstm, _ = self.lstm_backward(torch.stack(lstm_b))
        # Feed temporal outputs of CNN into LSTM
        module_outputs.append(b_lstm[-1])
        
        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm_forward(torch.stack(lstm_f))
        module_outputs.append(i_lstm[-1])
        
        # Concatenate current image CNN output 
        # and LSTMs output.
        x_cat = torch.cat(module_outputs, dim=-1)
        
        # Feed concatenated outputs into the 
        # regession networks.
        x_steer = self.control_angle(x_cat)
        if self.num_classes>1:
            x_steer = nn.Softmax()(x_steer)
        prediction = {'canSteering': torch.squeeze(x_steer),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction

class BiGRU_DO(nn.Module):
    def __init__(self, num_cnn_features=512, num_lstm_layers=3, hidden_lstm_size=64, num_classes=3, p_dropout=0.1):
        super(BiGRU_DO, self).__init__()
        self.num_classes = num_classes
        final_concat_size = 0
        # Main CNN
        # cnn = models.resnet34(pretrained=True)
        # cnn = ResNet34()
        # set_parameter_requires_grad(cnn, True)
        # self.features = nn.Sequential(*list(cnn.children())[:-1])
        
        # self.features = cnn
        self.intermediate = nn.Sequential(
            nn.Linear(num_cnn_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        final_concat_size += 128
        
        self.intermediate_2 = nn.Sequential(
            nn.Linear(num_cnn_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Forward LSTM
        self.lstm_forward = nn.GRU(input_size=128,
                            hidden_size=hidden_lstm_size,
                            num_layers=num_lstm_layers,
                            batch_first=False)
        final_concat_size += hidden_lstm_size
        
        # Backward LSTM
        self.lstm_backward = nn.GRU(input_size=128,
                            hidden_size=hidden_lstm_size,
                            num_layers=num_lstm_layers,
                            batch_first=False)
        final_concat_size += hidden_lstm_size
        
        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(32, num_classes),
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, data, hidden_features=None):
        module_outputs = []
        lstm_f = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        for k, v in data['cameraFront'].items():
            tmp_output = []
            for i in range(len(hidden_features)):
                tmp_output.append(hidden_features[i][k,])
            x_2 = torch.cat(tmp_output, 1)
            x = self.intermediate(x_2)
            lstm_f.append(x)
            if k == 0:
                module_outputs.append(self.intermediate_2(x_2))
                
        # reverse the list of front images
        lstm_b = lstm_f[::-1]
        # feed the reversed images into the lstm layer
        b_lstm, _ = self.lstm_backward(torch.stack(lstm_b))
        # Feed temporal outputs of CNN into LSTM
        module_outputs.append(b_lstm[-1])
        
        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm_forward(torch.stack(lstm_f))
        module_outputs.append(i_lstm[-1])
        
        # Concatenate current image CNN output 
        # and LSTMs output.
        x_cat = torch.cat(module_outputs, dim=-1)
        
        # Feed concatenated outputs into the 
        # regession networks.
        x_steer = self.control_angle(x_cat)
        if self.num_classes>1:
            x_steer = nn.Softmax()(x_steer)
        prediction = {'canSteering': torch.squeeze(x_steer),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction