# create a class LMC with single input and single output
import numpy as np

class LMCneuron:
    def __init__(self, inputs, vL, g):
        self.s = inputs
        self.vL = vL # resting potential
        self.g = g # conductance
        self.v = None # outputs
        self.alpha = 0.1 # scaling factor for calcium response


    def neuron_out(self):
        self.v = np.zeros_like(self.s) # outputs
        self.v[0] = self.vL
        for t in range(1, len(self.s)):
            dv_dt = -(self.v[t-1] - self.vL) + (1 / self.g) * self.s[t-1]
            self.v[t] = self.v[t-1] + dv_dt

            # Calcium response
            x = self.v[t] #+ (1 / self.g)*self.s[t-1]  # Input to rectifier
            self.v[t] = self.alpha * np.maximum(self.vL, x)
        return self.v
