
IMAGE_SIZE = 32
IMAGE_CHANNEL = 3

CIFAR_10_MEAN = [83.88608, 83.88608, 83.88608]
CIFAR_10_STD = [68.15831, 68.40918, 70.49192]

'''
0 : airplane
1 : automobile
2 : bird
3 : cat
4 : deer
5 : dog
6 : frog
7 : horse
8 : ship
9 : truck
'''
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASSES = len(CLASS_NAMES)

EMA_DECAY = 0.999
INIT_LEARNING_RATE = 0.002

WEIGHT_DECAY = 0.02 * INIT_LEARNING_RATE

MAX_ITERATION = 200000
SAVE_ITERATION = 5000
DECAY_ITERATION = [int(MAX_ITERATION * 0.5), int(MAX_ITERATION * 0.75)]

BATCH_SIZE = 64
