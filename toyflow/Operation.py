# coding: utf-8

import numpy as np
from queue import Queue

class Operation:
    def __init__(self, *ipt_nodes, name=None):
        self.ipt_nodes = ipt_nodes
        self.opt_nodes = []
        self.opt_value = None
        self.name = name
        self.graph = DEFAULT_GRAPH

        for n in ipt_nodes:             # so current node will be added as output in last node 
            n.opt_nodes.append(self)

        self.graph.operations.append(self)

    def compute_opt(self):
        pass

    def compute_gradient(self, grad=None):
        pass

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

class Add(Operation):
    """docstring for Add"""
    def __init__(self, x, y, name=None):
        super(Add, self).__init__(x, y, name=name)
    
    def compute_opt(self):
        x, y = self.ipt_nodes
        self.opt_value = np.add(x.opt_value, y.opt_value)
        return self.opt_value

    def compute_gradient(self, grad=None):
        vx, vy = [n.opt_value for n in self.ipt_nodes]
        if grad is None:
            grad = np.ones_like(self.opt_value)
        grad_wx = grad
        while np.ndim(grad_wx) > len(np.shape(vx)):
            grad_wx = np.sum(grad_wx, axis = 0)
        for axis, size in enumerate(np.shape(vx)):
            if size == 1:
                grad_wx = np.sum(grad_wx, axis = axis, keepdims=True)
        grad_wy = grad
        while np.ndim(grad_wy) > len(np.shape(vy)):
            grad_wy = np.sum(grad_wy, axis = 0)
        for axis, size in enumerate(np.shape(vy)):
            if size == 1:
                grad_wy = np.sum(grad_wy, axis = axis, keepdims=True)
        return [grad_wx, grad_wy]

def add(x, y, name=None):
    ''' Returns x + y
    '''
    return Add(x, y, name)

class Negative(Operation):
    """docstring for Negative"""
    def __init__(self, x, name=None):
        super(Negative, self).__init__(x, name=name)

    def compute_opt(self):
        x = self.ipt_nodes[0]
        self.opt_value = -x.opt_value
        return self.opt_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.opt_value)
        return -grad

class Multiply(Operation):
    """docstring for Multiply"""
    def __init__(self, x, y, name=None):
        super(Multiply, self).__init__(x, y, name=name)

    def compute_opt(self):
        x, y = self.ipt_nodes
        self.opt_value = x.opt_value * y.opt_value
        return self.opt_value

    def compute_gradient(self, grad=None):
        vx, vy = [n.opt_value for n in self.ipt_nodes]
        if grad is None:
            grad = np.ones_like(self.opt_value)
        grad_wx = grad
        while np.ndim(grad_wx) > len(np.shape(vx)):
            grad_wx = np.sum(grad_wx, axis = 0)
        for axis, size in enumerate(np.shape(vx)):
            if size == 1:
                grad_wx = np.sum(grad_wx, axis = axis, keepdims=True)
        grad_wy = grad
        while np.ndim(grad_wy) > len(np.shape(vy)):
            grad_wy = np.sum(grad_wy, axis = 0)
        for axis, size in enumerate(np.shape(vy)):
            if size == 1:
                grad_wy = np.sum(grad_wy, axis = axis, keepdims=True)
        return [grad_wx, grad_wy]

def multiply(x, y, name=None):
    ''' Returns x * y
    '''
    return Multiply(x, y, name)

class Matmul(Operation):
    """
        Matmul by numpy
    """
    def __init__(self, x, y, name=None):
        super(Matmul, self).__init__(x, y, name=None)

    def compute_opt(self):
        x, y = self.ipt_nodes
        self.opt_value = np.matmul(x.opt_value, y.opt_value)
        return self.opt_value

    def compute_gradient(self, grad=None):
        vx, vy = [n.opt_value for n in self.ipt_nodes]
        if grad is None:
            grad = np.ones_like(self.opt_value)
        dfdx = np.matmul(grad, np.transpose(vy))  # R = xy, dR/dx = grad*y.T
        dfdy = np.matmul(np.transpose(vx), grad)  # R = xy, dR/dx = x.T*grad
        return [dfdx, dfdy]

def matmul(x, y, name=None):
    ''' Multiplies matrix a by matrix b
    '''
    return MatMul(x, y, name)

class Log(Operation):
    """docstring for Log"""
    def __init__(self, x, name=None):
        super(Log, self).__init__(x, name=Name)

    def compute_opt(self):
        x = self.ipt_nodes[0]
        self.opt_value = np.log(x.opt_value)
        return self.opt_value

    def compute_gradient(self, grad=None):
        x = self.ipt_nodes[0].opt_value
        if grad is None:
            grad = np.ones_like(self.opt_value)
        return grad/x

def log(x, name=None):
    ''' Computes the natural logarithm of x element-wise.
    '''
    return Log(x, name=name)

class Sigmoid(Operation):
    """docstring for Sigmoid"""
    def __init__(self, x, name=None):
        super(Log, self).__init__(x, name=Name)
                        
    def compute_opt(self):
        x = self.ipt_nodes[0]
        self.opt_value = 1/(1+np.exp(-x.opt_value))
        return self.opt_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.opt_value)
        return grad*self.opt_value*(1-self.opt_value)

def sigmoid(x, name=None):
    ''' Computes sigmoid of x
    '''
    return Sigmoid(x, name=name)

class ReduceSum(Operation):
    """docstring for ReduceSum"""
    def __init__(self, x, axis=None):
        super(ReduceSum, self).__init__(x)
        self.axis = axis
        
    def compute_opt(self):
        x = self.ipt_nodes[0]
        self.opt_value = np.sum(x.opt_value, self.axis) # add by dimention axis
        return self.opt_value

    def compute_gradient(self, grad=None):
        input_value = self.ipt_nodes[0].opt_value
        if grad is None:
            grad = np.ones_like(self.opt_value)
        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)

def reduce_sum(x, axis=None):
    ''' Computes the sum of elements across dimensions of a tensor.
    '''
    return ReduceSum(x, axis=axis)

class Square(Operation):
    """docstring for Square"""
    def __init__(self, x, name=None):
        super(Square, self).__init__(x, name=name)

    def compute_opt(self):
        x = self.ipt_nodes[0]
        self.opt_value = np.square(x.opt_value)
        return self.opt_value

    def compute_gradient(self, grad=None):
        x = self.ipt_nodes[0].opt_value
        if grad is None:
            grad = np.ones_like(self.opt_value)
        return grad*np.multiply(2.0, x)

def square(x, name=None):
    ''' Computes square of x .
    '''
    return Square(x, name=name)

class Variable:
    def __init__(self, value=None, name=None, trainble=True):
        self.value = value
        self.opt_nodes = []
        self.opt_value = None
        self.name = name
        self.trainble = trainble
        self.graph = DEFAULT_GRAPH

        self.graph.variables.append(self)
        self.graph.trainble_vars.append(self) if trainble else None

    def compute_opt(self):
        if self.opt_value is None:
            self.opt_value = self.value
        return self.opt_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

class Constant:
    def __init__(self, value=None, name=None):
        self.value = value
        self.opt_nodes = []
        self.opt_value = None
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.constants.append(self)

    def compute_opt(self):
        if self.opt_value is None:
            self.opt_value = self.value
        return self.opt_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

class Placeholder:
    def __init__(self, name=None):
        self.opt_nodes = []
        self.opt_value = None
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def compute_gradients(target_operation):
    """
       implementation of backpropagation
       Breadth-First Search of target operation node
    """
    grad_table = {}
    queue = Queue()
    visited = set()

    grad_table[target_operation] = np.ones_like(target_operation.opt_value)
    queue.put(target_operation)
    visited.add(target_operation)

    while not queue.empty():
        node = queue.get()

        # compute the gradient from node's output
        if node != target_operation:
            grad_w_opts = []
            for opt_n in node.opt_nodes:
                grad_w_opt_n = grad_table[opt_n]
                grad_w_opt = opt_n.compute_gradient(grad_w_opt_n)
                if len(opt_n.ipt_nodes) > 1:
                    ipt_n_idx = opt_n.ipt_nodes.index(node)
                    grad_w_opts.append(grad_w_opt[ipt_n_idx])
                else:
                    grad_w_opts.append(grad_w_opt)
            grad_table[node] = sum(grad_w_opts)

        if hasattr(node, 'ipt_nodes'):
            for n in node.ipt_nodes:
                if n not in visited:
                    visited.add(n)
                    queue.put(n)

    return grad_table

