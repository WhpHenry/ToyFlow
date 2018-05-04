# coding: utf-8

class Graph:
    def __init__(self):
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainble_vars = [], []

    def __enter__(self):        
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH

    def as_default(self):
        """
            set this as default graph
        """
        return self