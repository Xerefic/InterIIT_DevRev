import numpy as np
import pandas as pd

import os
from subprocess import Popen, PIPE, STDOUT
import glob
import sys
import time
import timeit
import random
import copy
import json
import pickle as pkl
import joblib

import re
import string
import unidecode
from ast import literal_eval

import tqdm

import itertools
import collections
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, List, Dict, Any, Optional, cast, Type, Callable

import spacy
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
pronounciations_regex = r'\s*\((((.*[\u0250-\u02AF]+)|(.*[\u02B0–\u02FF]+)|(.*[\u1D00–\u1D7F]+)|(.*[\u1D80–\u1DBF]+)|(.*[\uA700–\uA71F]+)|(.*[\u2070–\u209F]+))\/).*?\)'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer

# from elasticsearch import Elasticsearch
from retriv import SearchEngine
# from ranx import compare, evaluate, fuse, optimize_fuse, Qrels, Run

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForSequenceClassification, PreTrainedModel, PretrainedConfig, AutoModelForQuestionAnswering
from transformers import pipeline as transformers_pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.util import semantic_search
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import SentenceEvaluator
from datasets import load_dataset

import onnxruntime
from optimum.onnxruntime import ORTOptimizer, ORTModelForQuestionAnswering
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.pipelines import pipeline as optimum_pipeline
import optuna
import onnx
from onnxsim import simplify as onnx_simplify
from onnxconverter_common import convert_float_to_float16


import logging
import warnings
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')