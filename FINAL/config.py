import torch
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/ukiyoe2photo"
VAL_DIR = "data/ukiyoe2photo"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_U = "genU.pth.tar"
CHECKPOINT_GEN_P = "genP.pth.tar"
CHECKPOINT_DISC_U = "discU.pth.tar"
CHECKPOINT_DISC_P = "discP.pth.tar"

transformU = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5261, 0.5975, 0.6413], std=[0.2350, 0.2589, 0.2811]),
        T.RandomVerticalFlip(p=.5)]
)
transformP = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.3926, 0.4092, 0.4123], std=[0.2778, 0.2486, 0.2710]),
        T.RandomVerticalFlip(p=.5)]
)