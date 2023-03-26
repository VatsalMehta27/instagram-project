import torch
from torch import nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, x):
        batch_size, time_steps, *input_shape = x.size()
        reshaped_x = x.view(-1, *input_shape)

        output = self.module(reshaped_x)

        # Reshape output tensor back to (batch_size, time_steps, *)
        output_shape = output.size()[1:]
        reshaped_output = output.view(batch_size, time_steps, *output_shape)

        return reshaped_output
