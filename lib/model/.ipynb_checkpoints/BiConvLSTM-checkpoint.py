# Code extends https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import torch.nn as nn
from torch.autograd import Variable
import torch


class BiConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(BiConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # NOTE: This keeps height and width the same
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(4 * self.hidden_dim))

        # TODO: we may want this to be different than the conv we use inside each cell
        self.conv_concat = nn.Conv2d(in_channels=2*self.hidden_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BiConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, sq_len,
                 bias=True, return_all_layers=False):
        super(BiConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        #hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        #if not len(kernel_size) == len(hidden_dim) == num_layers:
         #   raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, sq_len):
            cell_list.append(BiConvLSTMCell(input_size=(self.height, self.width),
                                            input_dim=self.input_dim,
                                            hidden_dim=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            bias=self.bias)
                             )
        self.cell_list = nn.ModuleList(cell_list)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        hidden_state = self._init_hidden(inpu=input_tensor)

        layer_output_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        backward_states = []
        forward_states = []
        output_inner = []


        for t in range(seq_len):
            hb, cb = hidden_state[0][0]
            hb, cb = self.cell_list[t](input_tensor=cur_layer_input[:, \
                                        seq_len - t - 1, :, :, :], cur_state=[hb, cb])
            backward_states.append(hb)

        for t in range(seq_len):
            hf, cf = hidden_state[0][1]
            hf, cf = self.cell_list[t](input_tensor=\
                                    cur_layer_input[:, t, :, :, :], cur_state=[hf, cf])
            forward_states.append(hf)

        for t in range(seq_len):
            h = self.cell_list[t].conv_concat(torch.cat((forward_states[t],\
                                                backward_states[seq_len - t - 1]), dim=1))
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)
        layer_output_list.append(layer_output)

        if not self.return_all_layers:
            return (1.5 * self.sigmoid(layer_output_list[-1][:, -1, :, :, :]) + 0.5)

        return layer_output_list

    def _init_hidden(self, inpu):
        init_states = []
        batch_size = inpu.size(0)
        device = inpu.device
        for i in range(1):
            init_states.append(((Variable(torch.zeros(batch_size, 1, self.height, self.width).to(device)),
                                Variable(torch.zeros(batch_size, 1, self.height, self.width).to(device))),
                                (Variable(torch.zeros(batch_size, 1, self.height, self.width).to(device)),
                                 Variable(torch.zeros(batch_size, 1, self.height, self.width).to(device)))
                                ))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param