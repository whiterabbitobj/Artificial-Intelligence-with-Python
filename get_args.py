import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Train a simple neural network to recognize pictures of flowers.",
            usage="EXAMPLE COMMAND:\npython train.py -a densenet169 -s saves -e 10 -hs 1000 500 -lr 0.001 -pe 3 -v 3 --gpu")
    parser.add_argument("-a", "--arch",
            help="The name of the pre-trained model to use. Currently supported: densenet121, densenet169, vgg16. Default: 'densenet169'.",
            type=str,
            default="densenet169")
    parser.add_argument("-b", "--batch_size",
            help="Batch size of the dataloaders used in training/testing. Default: 64",
            type=int,
            default=64)
    parser.add_argument("-c", "--crop_size",
            help="Crop size of dataset images. Default: 224",
            type=int,
            default=224)
    parser.add_argument("-dir", "--data_dir",
            help="Base directory of dataset images. Default: flowers/",
            type=str,
            default="flowers")
    parser.add_argument("-dr", "--drop_rate",
            help="Dropout rate of the classifier network. Default 0.5",
            type=float,
            default=0.5)
    parser.add_argument("-e", "--epochs",
            help="How many epochs to run through the training. Default: 10",
            type=int,
            default=10)
    parser.add_argument("--gpu",
            help="Run on the gpu using cuda, defaults to false",
            action="store_true")
    parser.add_argument("-hs", "--hidden_sizes",
            help="The size of the hidden layers for the classifier. Default: 1000 500.",
            nargs="+",
            type=int,
            default=[1000,500])
    parser.add_argument("-l", "--loaders",
            help="The folder names of the different datasets. e.g. train, valid, test. Default: train valid test",
            type=str,
            default=['train','valid','test'])
    parser.add_argument("-lr", "--learning_rate",
            help="The learning rate of the optimizer. Default: 0.001",
            type=float,
            default=0.001)
    parser.add_argument("--mean",
            help="The average (mean) of the dataset images. Default: 0.485, 0.456, 0.406",
            nargs="+",
            type=int,
            default=[0.485, 0.456, 0.406])
    parser.add_argument("-pe", "--print_every",
            help="How many batches between status prints. Default: 3",
            type=int, default=3)
    parser.add_argument("-rr", "--rand_rot",
            help="Random rotation size added to training images. Default: 30",
            type=int,
            default=30)
    parser.add_argument("-rs", "--rescale_size",
            help="Rescale before crop of testing images. Default: 256",
            type=int,
            default=256)
    parser.add_argument("-s", "--save_dir",
            help="Directory to save model checkpoints",
            type=str,
            default="")
    parser.add_argument("--std",
            help="Standard deviation of dataset images. Default: 0.229 0.224 0.225",
            nargs="+",
            type=int,
            default=[0.229, 0.224, 0.225])
    parser.add_argument("-tr", "--testrun",
            help="How many batches to run through before quitting. For testing purposes. Default: None",
            type=int,
            default=None)
    parser.add_argument("-v", "--do_validation",
            help="How often to run a validation, as a multiplier of the print_every flag. Higher numbers means less validation runs.\
                  1 means to run every time the status is printed. 0 will not run validation. Default: None",
            type=int,
            default=None)

    return parser.parse_args()



def get_predict_args():
    parser = argparse.ArgumentParser(description="Utilize a pre-trained model/classifier to predict the species of flower image in the input.",
            usage="\nEXAMPLE COMMAND:\npython predict.py -c checkpoint_densenet169_v1.pth -i flowers/test/11/image_03151.jpg -cat cat_to_name.json -k 3 --gpu" )
    parser.add_argument("-c", "--checkpoint",
            help="Path to the model checkpoint to use for predictions.",
            type=str,
            default=None)
    parser.add_argument("-cat", "--category_names",
            help="Path to .json file where the name/index pairs are stored",
            type=str,
            default="cat_to_name.json")
    parser.add_argument("-i", "--input",
            help="Path to flower you wish to classify",
            type=str,
            default="")
    parser.add_argument("--gpu",
            help="Run on the gpu using cuda, defaults to false",
            action="store_true")
    parser.add_argument("-k", "--topk",
            help="How many predictions to return. The top 'K' results.",
            type=int,
            default=5)
    parser.add_argument("-rand", "--random",
            help="Overrides 'input' argument. Returns results for a random image\
                  in the provided directory. Recursive, will randomly select from\
                  all subdirs in the provided directory.",
            type=str,
            default='')
    parser.add_argument("-show", "--show_image",
            help="Will display an image and graph if this flag is called.",
            action="store_true")

    return parser.parse_args()
