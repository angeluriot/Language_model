import os, torch
from contextlib import nullcontext

# ============== Dataset ============== #

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
NUM_THREADS = 16

TOKENIZER_DATA_SIZE = 300_000_000
MIN_DOCUMENT_SIZE = 64
PRETRAINING_VAL_RATIO = 0.001
MAX_TOKEN_LENGTH = 16
CONTROL_TOKENS = ['⮜unknown⮞', '⮜padding⮞', '⮜start-of-text⮞', '⮜tab⮞', '⮜new-line⮞', '⮜human⮞', '⮜system⮞', '⮜user⮞', '⮜assistant⮞', '⮜end-of-text⮞']
PADDING_TOKEN = 1
FORCED_TOKENS = ['Dimension', ' Dimension', 'GPT', ' GPT', 'IA', ' IA', 'Generative', ' Generative', 'Pre', ' Pre', 'trained', ' trained', 'Transformer', ' Transformer']

FINETUNING_VAL_RATIO = 0.01

SPLIT_RATIOS = [
	0.099,	# human
	0.9,	# chatbot
	0.001	# DimensionGPT
]

HUMAN_PREPROMPT_RATIOS = [
	0.3,	# human
	0.0,	# chatbot
	0.0,	# DimensionGPT
	0.7		# None
]

CHATBOT_PREPROMPT_RATIOS = [
	0.0,	# human
	0.5,	# chatbot
	0.4,	# DimensionGPT
	0.1		# None
]

DIMENSION_GPT_PREPROMPT_RATIOS = [
	0.0,	# human
	0.0,	# chatbot
	1.0,	# DimensionGPT
	0.0		# None
]

INSTRUCTION_LOSS_STRENGTH = 0.1
PREPROMPT = "Une discussion entre un utilisateur et DimensionGPT, un modèle de langage conversationnel français créé par le développeur indépendant Dimension et basé sur l'architecture GPT."

# =============== Model =============== #

VOCAB_SIZE = 32_000
MAX_CONTEXT = 512
WINDOW_SIZE = 256
EMBEDDING_DIM = 1024
NUM_GROUPED_HEADS = 4
NUM_HEADS = 16
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
FFN_DIM = int((2.0 / 3.0) * 4 * EMBEDDING_DIM)
NUM_BLOCKS = 16
DROPOUT = 0
INIT_STDDEV = 0.02
ROPE_THETA = 10000.0

# ============= Training ============== #

BATCH_SIZE = 16
NUM_ACCUMULATIONS = 64

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_STEPS = 2_000
DECAY_STEPS = 100_000

BETA_1 = 0.9
BETA_2 = 0.95
EPSILON = 1e-5
WEIGHT_DECAY = 0.1
CLIP_GRADIENT = 1.0

METRICS_BETA = 0.9
VAL_INTERVAL = 50

# ===================================== #

GPU_ENABLED = torch.cuda.is_available()
FLOAT16_ENABLED = GPU_ENABLED and torch.cuda.is_bf16_supported()
DEVICE_NAME = 'cuda:0' if GPU_ENABLED else 'cpu'
DEVICE = torch.device(DEVICE_NAME)
CONTEXT = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if FLOAT16_ENABLED else nullcontext()
