import os, torch

# ============== Dataset ============== #

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
NUM_THREADS = 16
VAL_RATIO = 0.001
TOKENIZER_DATA_SIZE = 300_000_000
SAVING_BATCHS = 1024

CC100_NB_DOCUMENTS = 59_448_891
CC100_NB_CHARS = 141_796_063_805

WIKIPEDIA_NB_DOCUMENTS = 2_402_095
WIKIPEDIA_NB_CHARS = 6_916_728_196

FRENCH_REDDIT_PATH = 'E:/Angel/Development/Datasets/French_reddit/dataset.xml'
FRENCH_REDDIT_NB_DOCUMENTS = 556_621
FRENCH_REDDIT_NB_CHARS = 445_808_054

CONTROL_CHARS = ['↲', '␃', '␄', '�']

# =============== Model =============== #

VOCAB_SIZE = 30_000
MAX_CONTEXT = 256
EMBEDDING_DIM = 768
NUM_HEADS = 12
FFN_DIM = 4 * EMBEDDING_DIM
NUM_BLOCKS = 12
DROPOUT = 0.0
USE_BIAS = False
INIT_STDDEV = 0.02

# ============= Training ============== #

BATCH_SIZE = 16
NUM_ACCUMULATIONS = 16

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_STEPS = 1_000
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


DONT_KNOW_ANSWERS = [
	"Apologies , I 'm unable to comprehend the query .",
	"Forgive me , but the question is unclear to me .",
	"I 'm afraid I do n't grasp the question .",
	"Pardon me , I 'm not able to make sense of the inquiry .",
	"Regrettably , I 'm struggling to understand the question .",
	"I apologize , but the query confuses me .",
	"Excuse me , I 'm having difficulty deciphering the question .",
	"Sorry , the question is n't clear to me .",
	"My apologies , the query seems ambiguous .",
	"Unfortunately , I cannot fathom the question .",
	"I beg your pardon , but I do n't get the question .",
	"Regretfully , the inquiry eludes me .",
	"Forgive my confusion , but I do n't understand the question .",
	"I 'm sorry , but the question seems perplexing to me .",
	"My apologies , I 'm finding it hard to comprehend the query .",
	"Pardon my misunderstanding , but I don't grasp the question .",
	"Sorry , but the question is rather puzzling to me .",
	"Please excuse me , as I 'm unable to decipher the inquiry .",
	"I 'm afraid I ca n't quite make out the question .",
	"Apologies , but I 'm struggling with the query .",
	"I 'm sorry , the question appears to be unclear .",
	"Regrettably , I 'm not able to grasp the question .",
	"Pardon my confusion , I ca n't comprehend the inquiry .",
	"Forgive me , but the question is beyond my understanding .",
	"My apologies , but I 'm having trouble with the question .",
	"I 'm afraid I ca n't quite grasp the inquiry .",
	"Excuse me , I do n't understand the question posed .",
	"Sorry , I 'm unable to make sense of the question .",
	"I beg your pardon , but the query is difficult for me to understand .",
	"Regretfully , I ca n't seem to comprehend the question .",

	"Apologies , I 'm unaware of the solution to this query .",
	"My apologies , I 'm unable to provide an answer to this question .",
	"Regrettably , I cannot offer a response to this inquiry .",
	"I 'm afraid I lack the knowledge to address this question .",
	"Pardon me , but I do n't have the information to answer this question .",
	"Forgive me , I 'm not familiar with the answer to this query .",
	"I 'm sorry , but I do n't possess the knowledge to respond to this question .",
	"Unfortunately , I 'm unable to give a satisfactory answer to this inquiry .",
	"I apologize , but I do n't know the solution to this problem .",
	"My regrets , I ca n't supply a response to this particular question .",
	"Excuse me , but I am uninformed about the answer to this matter .",
	"I wish I could help , but I do n't know the answer to this question.",
	"I 'm sorry , but I ca n't provide any insight on this topic .",
	"Please forgive my ignorance , but I do n't know the answer to this question .",
	"Regretfully , I 'm not able to provide a response to this inquiry .",
	"I 'm afraid I do n't have the expertise to answer this question .",
	"Apologies , but I 'm at a loss when it comes to answering this question .",
	"Unfortunately , I do n't have the required knowledge to address this query .",
	"I 'm sorry , but I ca n't seem to find a solution to this question .",
	"Please pardon me , I lack the information to respond to this inquiry .",
	"I 'm sorry , but I 'm unable to provide a response to this particular matter .",
	"I regret to inform you that I do n't know the answer to this question .",
	"Unfortunately , I ca n't seem to provide an answer to this query .",
	"I 'm afraid I 'm not equipped to address this question .",
	"Excuse my lack of knowledge , but I do n't know the answer to this question .",
	"My apologies , but I 'm at a loss for an answer to this inquiry .",
	"Regrettably , I 'm not capable of providing a solution to this question .",
	"I 'm sorry , but I do n't have the necessary information to answer this query .",
	"Please forgive me , but I 'm unable to provide a response to this question ."
]
