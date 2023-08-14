import os, shutil, random
import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from tqdm import tqdm

from gpt.data.tokenizer import Tokenizer
from gpt.settings import *
from gpt import utils
from gpt.data import shared


def import_dataset():

	dataset = load_dataset('mc4', 'fr', num_proc = NUM_THREADS)

