import numpy as np
import pandas as pd

import os
import glob
import time
import random
import json
import copy
import tqdm
import pyprind
import itertools
import pickle as pkl
from dataclasses import dataclass, field
from typing import Union, List, Dict, Any, Optional, cast

import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from datasets import load_dataset