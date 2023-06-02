import numpy as np
import numpy.typing as npt

from gpt.settings import *
from gpt.data.tokenizer import Tokenizer
from gpt.data import CC100


def get_data() -> tuple[Tokenizer, npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:

	if DATASET == 'CC100':
		return CC100.get_data()
