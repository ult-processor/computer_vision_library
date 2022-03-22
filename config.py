import torch

BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 12 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory

TRAIN_DIR = '../waste/train'
# validation images and XML files directory
VALID_DIR = '../waste/test'

# classes: 0 index is reserved for background

# classes: 0 index is reserved for background
CLASSES = [
    'pitol', 'knife'
]
NUM_CLASSES = 2


# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
MODEL_DICT_FILE = OUT_DIR + '/model12.pth'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs