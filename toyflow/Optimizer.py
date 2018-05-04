# coding: utf-8

from .Operation import Operation, compute_gradients

class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss_op): 
        
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute_opt(self):
                grad_table = compute_gradients(loss_op)
                # Iterate all trainable variables in graph.
                for var in DEFAULT_GRAPH.trainble_vars:
                    if var in grad_table:
                        grad = grad_table[var]
                    # Update its output value.
                    var.opt_value -= learning_rate*grad

        return MinimizationOperation()