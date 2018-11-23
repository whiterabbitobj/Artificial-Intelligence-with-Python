import os
import torch
from torch import optim
from torchvision import  datasets, transforms, models
from torch.utils.data import DataLoader
from random import randint
from model_data import Network

############################
# LOADING FLOWER DATA IMAGES
############################

def get_random_flower_path(basepath):
    flowers = []
    for path, folders, files in os.walk(basepath):
        for file in files:
            flowers.append(os.path.join(path,file))
    return flowers[randint(0,len(flowers)-1)]


def load_datasets(mean=[0.485, 0.456, 0.406],
                  std = [0.229, 0.224, 0.225],
                  crop_size = 224,
                  rescale_size = 256,
                  batch_size = 64,
                  rand_rot = 30,
                  data_dir="flowers",
                  loaders=["train","valid","test"]
                  ):
    training_transforms = transforms.Compose([transforms.RandomRotation(rand_rot),
                                   transforms.RandomResizedCrop(crop_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    testing_transforms = transforms.Compose([transforms.Resize(rescale_size),
                                   transforms.CenterCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])

    loader_transforms = {loaders[0]:training_transforms}
    for loader in loaders[1:]:
        loader_transforms[loader] = testing_transforms
    loader_datasets = {loader:datasets.ImageFolder(data_dir+'/'+loader, transform=loader_transforms[loader]) for loader in loaders}
    dataloaders = {}
    for loader in loaders:
        do_shuffle = (loader=='train')
        dataloaders[loader] = DataLoader(loader_datasets[loader], batch_size=batch_size, shuffle=do_shuffle)
    return dataloaders, loader_datasets['train'].class_to_idx



################################
# SAVING AND LOADING CHECKPOINTS
################################

def save_checkpoint(model, save_name):
    '''
        Pass in a model with the appropriate sub attributes and a filepath for saving.
        Requires these to be defined in the model:
        model.optimizer_sate
        model.classifier.epochs
    '''
    checkpoint = {'arch': model.name,
                  'class_to_idx': model.class_to_idx,
                  'input_size': model.classifier.hidden_sizes[0].in_features,
                  'output_size': model.classifier.output.out_features,
                  'hidden_layers': [layer.out_features for layer in model.classifier.hidden_sizes],
                  'drop_rate': model.classifier.dropout.p,
                  'epochs': model.classifier.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': model.optimizer_state,
                  'lr':model.lr}
    model.to('cpu')
    torch.save(checkpoint, save_name)
    return True



def load_checkpoint(filepath):
    '''Loads a checkpoint from an earlier trained model.
        Requires these custom sub-attributes to be present in the save file:
        arch
        optimizer
        epochs
    '''

    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = checkpoint['arch']
    elif checkpoint['arch'] == 'densenet169':
        model = models.densenet169(pretrained=True)
        model.name = checkpoint['arch']
    elif checkpoint['arch']  == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Sorry, this checkpoint asks for a model that isn't supported! ({})".format(checkpoint['arch']))

    model.classifier = Network(checkpoint['input_size'],
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'],
                               checkpoint['drop_rate']
                               )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.lr = checkpoint['lr']

    optimizer = optim.Adam(model.classifier.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model
