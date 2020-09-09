# Image channels
CHANNELS = 3

# Hidden layers
GEN_HIDDEN = 64
DIS_HIDDEN = 64

# Residual blocks
RES_BLOCKS = 9

# Image size
IMG_SIZE = 256

BATCH_SIZE = 1 # because we use instantnorm not batchnorm
EPOCHS = 25

LR = 2e-5
BETAS = (0.888, 0.999)
