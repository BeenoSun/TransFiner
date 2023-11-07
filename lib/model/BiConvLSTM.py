# Code extends https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

# Code extends https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import torch.nn as nn
from torch.autograd import Variable
import torch
from .networks.dla import DLASeg, DeformConv
from torch.cuda.amp import autocast
from opts import opts
opt = opts().parse()
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

    @autocast(enabled=opt.withamp)
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
                 opt=None, bias=True, return_all_layers=False):
        super(BiConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        #hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        #if not len(kernel_size) == len(hidden_dim) == num_layers:
         #   raise ValueError('Inconsistent list length.')
        self.opt = opt
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
        self.tanh = nn.Tanh()
        # self.up_scale4x = nn.Sequential(
        #         # upsample part
        #         nn.Conv2d(1, 3, 1, 1),
        #         nn.Conv2d(3, 3, 1, 1),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
        #         )
        #head['hm_pr'] = 1
        #head_conv['hm_pr'] = [256]
        self.hm_corr_network = DLASeg(34, heads={'hm':1, 'hm_pr':1, 'wh':2},
            head_convs={'hm':[256], 'hm_pr':[256], 'wh':[256]}, opt=opt)
        
        self.chan_weig_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)))
        self.chan_weig_lin = nn.Sequential(
            nn.Linear(1536, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(768, 1536, bias=False),
            nn.Sigmoid()
        )

        self.lstm_aggre_defor = nn.Sequential(
            DeformConv(408, 408),
            DeformConv(408, 408),
            DeformConv(408, 408),
            DeformConv(408, 128),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        )
        
        self.lstm_aggre = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
        )
        '''
        self.aggre_frac_resi = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 1,
                      kernel_size=(self.hidden_dim * sq_len, 1, 1),
                      padding=(0, 0, 0),
                      stride=(1, 1, 1),
                      bias=self.bias),
            nn.BatchNorm2d(1),
            nn.ReLU())
            '''
    @autocast(enabled=opt.withamp)
    def forward(self, input_sec):
        # bb_input = self.up_scale4x(layer_output_list[-1][:, -1, :, :, :])
        hm_refine, _ = self.hm_corr_network(input_sec)
        # y = tanh((x-1.5*tanh(0.7*x)));
        #hm_refine[0]['hm'] = self.tanh(hm_refine[0]['hm'] - 1.5*self.tanh(0.7*hm_refine[0]['hm']))
        #hm_refine[0]['hm'] = self.tanh(hm_refine[0]['hm_pr'])
        '''
        # double buffer
        #y = tanh(7*(x(x>=0)-1.5*tanh(0.68*x(x>=0))));
        #y = tanh(7*(-abs(-x(x<0)+1.5*tanh(0.68*x(x<0)))));
        x_pos = hm_refine[0]['hm'] >= 0
        x_neg = hm_refine[0]['hm'] < 0
        hm_refine[0]['hm'][x_pos] = \
          self.tanh(7.*(hm_refine[0]['hm'] - 1.48 * self.tanh(0.68 * hm_refine[0]['hm'])))[x_pos]
        hm_refine[0]['hm'][x_neg] = \
          self.tanh(7.*(-torch.abs(-hm_refine[0]['hm']+1.48*self.tanh(0.68 * hm_refine[0]['hm']))))[x_neg]
        hm_refine[0]['hm_pr'] = 1.4 * self.sigmoid(hm_refine[0]['hm_pr']) - 0.55
        '''
        # new lstmv2 model
        '''
        input_tensor = input_tensor.view(input_tensor.shape[0], -1, input_tensor.shape[3], input_tensor.shape[4]).contiguous()
        weig = self.chan_weig_pool(input_tensor).squeeze(2).squeeze(2)
        weig = self.chan_weig_lin(weig).unsqueeze(2).unsqueeze(2)
        inp = input_tensor * weig
        inp = self.lstm_aggre(inp)
        hm_refine, _ = self.hm_corr_network(inp)
        hm_refine[0]['hm'] = self.opt.weight * self.tanh(hm_refine[0]['hm']) + self.opt.bias
        hm_refine[0]['hm_pr'] = self.sigmoid(hm_refine[0]['hm_pr']) - 0.45
        #hm_corr = self.hm_corr_network(img, pre_hm=layer_output_list[-1][:, -1, :, :, :])'''
        return self._sigmoid(hm_refine[0]['hm']), self._sigmoid(hm_refine[0]['hm_pr']), hm_refine[0]['wh']


    def _init_hidden(self, inpu):
        init_states = []
        batch_size = inpu.size(0)
        device = inpu.device
        for i in range(1):
            init_states.append(((Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)),
                                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))),
                                (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)),
                                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)))
                                ))
        return init_states

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

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

'''

note: second input with feats (second round v2)

import torch.nn as nn
from torch.autograd import Variable
import torch
from .networks.dla import DLASeg
from torch.cuda.amp import autocast
from opts import opts
opt = opts().parse()
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
                 opt=None, bias=True, return_all_layers=False):
        super(BiConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        #hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        #if not len(kernel_size) == len(hidden_dim) == num_layers:
         #   raise ValueError('Inconsistent list length.')
        self.opt = opt
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
        self.tanh = nn.Tanh()
        # self.up_scale4x = nn.Sequential(
        #         # upsample part
        #         nn.Conv2d(1, 3, 1, 1),
        #         nn.Conv2d(3, 3, 1, 1),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
        #         )
        #head['hm_pr'] = 1
        #head_conv['hm_pr'] = [256]
        self.hm_corr_network = DLASeg(34, heads={'hm':1, 'hm_pr':1},
            head_convs={'hm':[256], 'hm_pr':[256]}, opt=opt)
        self.chan_weig_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)))
        self.chan_weig_lin = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64, bias=False),
            nn.Sigmoid()
        )
        self.lstm_aggre = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        )
        self.aggre_frac_resi = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 1,
                      kernel_size=(self.hidden_dim * sq_len, 1, 1),
                      padding=(0, 0, 0),
                      stride=(1, 1, 1),
                      bias=self.bias),
            nn.BatchNorm2d(1),
            nn.ReLU())
    @autocast(enabled=opt.withamp)
    def forward(self, img, input_tensor):
        # bb_input = self.up_scale4x(layer_output_list[-1][:, -1, :, :, :])

        # new lstmv2 model
        weig = self.chan_weig_pool(input_tensor).squeeze(2).squeeze(2)
        weig = self.chan_weig_lin(weig).unsqueeze(2).unsqueeze(2)
        inp = input_tensor * weig
        inp = self.lstm_aggre(inp)
        hm_refine, _ = self.hm_corr_network(img, pre_hm=inp)
        hm_refine[0]['hm'] = self.opt.weight * self.tanh(hm_refine[0]['hm']) + self.opt.bias
        hm_refine[0]['hm_pr'] = self.sigmoid(hm_refine[0]['hm_pr']) / 1.2 - 0.6
        #hm_corr = self.hm_corr_network(img, pre_hm=layer_output_list[-1][:, -1, :, :, :])
        return hm_refine[0]['hm'], hm_refine[0]['hm_pr']


    def _init_hidden(self, inpu):
        init_states = []
        batch_size = inpu.size(0)
        device = inpu.device
        for i in range(1):
            init_states.append(((Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)),
                                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))),
                                (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)),
                                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device)))
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
'''