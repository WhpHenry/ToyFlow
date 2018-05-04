# coding: utf-8

from functools import reduce
from .Operation import Operation, Variable, Placeholder

class Session:

    def __init__(self):
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        """
            Context management protocal method called before with-block
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
            Context management protocal method called before with-block
        """
        self.close()

    def close(self):
        """
            clear all node value
        """
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainble_vars)
        for n in all_nodes:
            n.opt_value = None

    def run(self, operation, feed_dict=None):
        """
            postorder traversal of operation to calculate
        """
        post_nodes = _forward(operation)
        for n in post_nodes:
            if isinstance(n, Placeholder):
                n.opt_value = feed_dict[n]
            else:
                n.compute_opt()
        return operation.opt_value

def _forward(operation):
    """
        postorder traversal of operation to calculate
    """
    post_nodes = []
    def postorder_traverse(operation):
        if isinstance(operation, Operation):
            for ipt_node in operation.ipt_nodes:
                postorder_traverse(ipt_node)
        post_nodes.append(operation)        
    postorder_traverse(operation)
    return post_nodes