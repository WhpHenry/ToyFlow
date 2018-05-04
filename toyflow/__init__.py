# coding: utf-8
# forked from https://github.com/PytLab/simpleflow

from .Graph import *
from .Operation import *
from .Session import *
from .Optimizer import *

# Create a default graph.
import builtins
DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()