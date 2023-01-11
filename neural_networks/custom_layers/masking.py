# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import torch
from torch import nn

class Camouflage(nn.Module):
    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True

    def forward(self, inputs):
        if len(inputs[1].shape) == 3:
            boolean_mask = torch.any(torch.not_equal(inputs[1], self.mask_value),
                                 axis=-1, keepdims=True)
        else:
            boolean_mask = torch.expand_dims(torch.not_equal(inputs[1], self.mask_value))
        boolean_mask = boolean_mask.type_as(inputs[0])
        return inputs[0] * boolean_mask

    # def get_config(self):
    #     config = {'mask_value': self.mask_value}
    #     base_config = super(Camouflage, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
    #
    # def compute_output_shape(self, input_shape):
    #     return input_shape
    #
    # def compute_mask(self, inputs, mask=None):
    #     return None