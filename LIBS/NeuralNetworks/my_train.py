'''
  the difference between valid() and test() is that valid compute loss.
'''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from LIBS.NeuralNetworks.Helper.my_is_inception import is_inception_model


def train_regression(model, loader_train, criterion, optimizer, scheduler,
                     epochs_num, log_interval_train=10, log_interval_test=None,
                     show_detail_train=False, show_detail_test=False,
                     save_model_dir=None,
                     loader_valid=None, loader_test=None, accumulate_grads_times=None):

    is_inception = is_inception_model(model)
    if is_inception:
        model.AuxLogits.training = False
        num_filters = model.AuxLogits.fc.in_features
        num_class = model.last_linear.out_features
        model.AuxLogits.fc = nn.Linear(num_filters, num_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1 and (not is_inception):
        model = nn.DataParallel(model)

    for epoch in range(epochs_num):
        print(f'Epoch {epoch}/{epochs_num - 1}')
        model.train()
        running_loss, epoch_loss = 0, 0

        for batch_idx, (inputs, labels) in enumerate(loader_train):
            inputs, labels = inputs.to(device), labels.to(device)

            if is_inception:
                # inception v3 use an auxiliary loss function
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                # loss2 = criterion(aux_outputs, labels)
                # loss = loss1 + 0.4 * loss2
                loss = loss1
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if show_detail_train:
                y_predicts = outputs.detach().cpu().numpy()
                y_gt = labels.detach().cpu().numpy()

                for i in range(y_predicts.shape[0]):
                    print(f'pred:{int(y_predicts[i][0])},{int(y_predicts[i][1])},gt:{int(y_gt[i][0])},{int(y_gt[i][1])}',)

            if accumulate_grads_times is None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batch_idx % accumulate_grads_times == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # statistics
            #show average loss instead of batch total loss  *inputs.size(0) total loss
            running_loss += loss.item()
            epoch_loss += loss.item()

            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}] loss:{running_loss / log_interval_train:8.2f}')
                    running_loss = 0.0

        print(f'epoch{epoch} loss: {epoch_loss / (batch_idx+1):8.2f}')

        scheduler.step()

        if loader_valid:
            print('compute validation dataset...')
            validate(model, loader_valid, criterion, show_detail_test, log_interval_test)
        if loader_test:
            print('compute test dataset...')
            validate(model, loader_test, criterion, show_detail_test, log_interval_test)

        if save_model_dir:
            save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth')
            print('save model:', save_model_file)
            os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict, save_model_file)



def validate(model, dataloader, criterion, show_details=False, log_interval=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        epoch_loss, running_loss = 0, 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if show_details:
                y_predicts = outputs.detach().cpu().numpy()
                y_gt = labels.detach().cpu().numpy()
                for i in range(y_predicts.shape[0]):
                    print(f'pred:{int(y_predicts[i][0])},{int(y_predicts[i][1])},gt:{int(y_gt[i][0])},{int(y_gt[i][1])}',)

            running_loss += loss.item()
            epoch_loss += loss.item()

            if log_interval is not None:
                if batch_idx % log_interval == log_interval - 1:
                    print(f'[batch:{batch_idx}] loss:{running_loss / log_interval:8.3f}')
                    running_loss = 0.0

        print(f'loss:{ epoch_loss / (batch_idx+1):8.2f}')


