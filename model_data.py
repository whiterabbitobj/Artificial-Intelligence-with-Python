import os
import time
from random import randint
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
#from torchvision import datasets, transforms, models
#from torch.utils.data import DataLoader


##########################
# DEAL WITH MODEL CREATION
##########################

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[1000,500], drop_rate=0.5):
        ''' Builds a flexible class to add arbitrary complexity to the
            classifier network.
        '''
        super().__init__()

        # Add the first network layer
        self.hidden_sizes = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])

        # Add an arbitrary amount of additional network layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.hidden_sizes.extend([nn.Linear(i, o) for i, o in layer_sizes])

        # Add the final network layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)

        # Add the dropout rate
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        ''' Forward pass through the network, returns the output information, must use this
            class based function as there is no built-in forward() method associated with
            nn.ModuleList().
        '''

        # Calculate the output of each layer and apply the activation function (ReLU), then perform the dropout
        for linear in self.hidden_sizes:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        # Softmax the results of the network as the result of this forward() method
        return F.log_softmax(x, dim=1)



def load_model(model_name, hidden_sizes, drop_rate):
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = 'densenet121'
        in_features = model.classifier.in_features
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        model.name = 'densenet169'
        in_features = model.classifier.in_features
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
        in_features = model.classifier[0].in_features
    # Freeze the params of the pretrained network
    for p in model.parameters():
        p.requires_grad = False
    classifier = Network(input_size=in_features, output_size=102, hidden_sizes=hidden_sizes, drop_rate=drop_rate)
    model.classifier = classifier

    return model



######################################
# TESTING MODEL TRAINING WITH DATASETS
######################################

def validation(model, loader, criterion, device='cuda', do_time=True):
    test_loss = 0
    accuracy = 0
    start = time.time()
    model.eval()
    model.to(device)
    for images, classes in loader:
        images, classes = images.to(device), classes.to(device)

        output = model.forward(images)
        test_loss += criterion(output, classes).item()

        ps = torch.exp(output)
        equality = (classes.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    test_loss *= 1/len(loader)
    accuracy *= 1/len(loader)
    report = "..Test Loss: {:.4f} ..Test Accuracy: {:.4f}".format(test_loss, accuracy)
    if do_time:
        report += "\n...it took {:.2f} to run the validation test".format(time.time()-start)

    return report



def test_model_results(model, dataloaders, criterion, device='cuda'):

    model.eval()
    model.to(device)
    with torch.no_grad():
        accuracy = 0
        for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                # Run the test data through the model
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy = equality.type(torch.FloatTensor).mean()
                accuracy = accuracy.numpy()
                print(f'Loss: {loss.item():.3f}, Accuracy: {accuracy}')

    return
