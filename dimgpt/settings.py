import torch

# ============== Dataset ============== #

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
NUM_THREADS = 16
VAL_RATIO = 0.001
TOKENIZER_DATA_SIZE = 300_000_000
SAVING_BATCHS = 1024
CONTROL_CHARS = ['<tab>', '<nl>', '<sot>', '<som>', '<user>', '<bot>', '<eom>', '<eot>', '<unk>']
MAX_TOKEN_LENGTH = 16
AVERAGE_TOKEN_LENGTH = 4.25

# =============== Model =============== #

VOCAB_SIZE = 32_000
MAX_CONTEXT = 512
EMBEDDING_DIM = 768
NUM_HEADS = 12
FFN_DIM = 4 * EMBEDDING_DIM
NUM_BLOCKS = 12
DROPOUT = 0.0
USE_BIAS = False
INIT_STDDEV = 0.02

# ============= Training ============== #

BATCH_SIZE = 32
NUM_ACCUMULATIONS = 16

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_STEPS = 2_000
DECAY_STEPS = 500_000

WEIGHT_DECAY = 0.1
BETA_1 = 0.9
BETA_2 = 0.95
CLIP_GRADIENT = 1.0

METRICS_BETA = 0.9
VAL_INTERVAL = 50

# ===================================== #

GPU_ENABLED = torch.cuda.is_available()
DEVICE_NAME = 'cuda:0' if GPU_ENABLED else 'cpu'
DEVICE = torch.device(DEVICE_NAME)

# ===================================== #

# CC100 nb documents: 59,448,891
# CC100 nb chars: 141,796,063,805

# Wikipedia nb documents: 2,402,095
# Wikipedia nb chars: 6,916,728,196

# French instructs nb documents: 11,794,112
# French instructs nb chars: 5,604,589,425

# French reddit nb documents: 556,621
# French reddit nb chars: 445,808,054

# French tweets nb documents: 1,526,724
# French tweets nb chars: 119,159,266

# My tweets nb documents: 70,035
# My tweets nb chars: 4,638,984