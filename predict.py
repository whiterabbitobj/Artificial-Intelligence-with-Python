import os
import json
import numpy as np
import pandas as pd
import torch
from torchvision import models

from model_data import Network, load_model, validation
from utility import activate_device, process_image, show_prediction_results, print_prediction_results
from data_handling import load_checkpoint, load_datasets, get_random_flower_path
from get_args import get_predict_args

# EXAMPLE COMMAND
#python predict.py -c checkpoint_densenet169_v2.pth
# -i flowers/test/11/image_03151.jpg -cat cat_to_name.json -k 3 --gpu -show

def main():
    in_args = get_predict_args()
    model = load_checkpoint(in_args.checkpoint)
    active_device = activate_device(in_args.gpu)

    with open(in_args.category_names, 'r') as f:
        idx_to_name = json.load(f)

    if os.path.isdir(in_args.random):
        flower_path = get_random_flower_path(in_args.random)
    elif os.path.isfile(in_args.input):
        flower_path = in_args.input
    else:
        print("Requested file does not exist ({})".format(in_args.input))
        return

    print("{}\nChecking image located at: {}".format("#"*40, flower_path))

    flower_class= os.path.basename(os.path.dirname(flower_path))
    flower_label = idx_to_name[flower_class]

    df = predict(flower_path, model, active_device, in_args.topk, idx_to_name)
    print_prediction_results(df, flower_label)

    if in_args.show_image:
        show_prediction_results(df, flower_label, flower_path)

    return



def predict(image_path, model, active_device='cuda', topk=5, idx_to_name={}):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    #model.to(active_device)

    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    #image.to(active_device)
    image.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(image)

    top_probs, top_classes = torch.topk(output, topk)

    top_probs = torch.exp(top_probs).data.numpy()[0]
    top_classes = top_classes.data.numpy()[0]

    invert_idx = {v:k for k,v in model.class_to_idx.items()}
    indexes = [invert_idx[c] for c in top_classes]
    labels = [idx_to_name[idx].title() for idx in indexes]
    classes = [int(x) for x in top_classes]
    probs = [int(x*10000)/10000 for x in top_probs]
    df = pd.DataFrame(np.array([classes, probs]).transpose(1,0), index=labels, columns=['model class','probability'])
    return df



if __name__ == "__main__":
    main()
