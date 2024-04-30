import torch
import torch.nn.functional as F
from torch.distributions import Normal, StudentT
import numpy as np
from itertools import combinations

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def errors_ber(y_true, y_pred, mask=None):
    if mask == None:
        mask=torch.ones(y_true.size(),device=y_true.device)
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    mask = mask.view(mask.shape[0], -1, 1)
    myOtherTensor = (mask*torch.ne(torch.round(y_true), torch.round(y_pred))).float()
    res = sum(sum(myOtherTensor))/(torch.sum(mask))
    return res

def errors_bler(y_true, y_pred, get_pos = False):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    if not get_pos:
        return bler_err_rate
    else:
        err_pos = list(np.nonzero((np.sum(tp0,axis=1)>0).astype(int))[0])
        return bler_err_rate, err_pos

def corrupt_signal(input_signal, sigma = 1.0, noise_type = 'awgn', vv =5.0, radar_power = 20.0, radar_prob = 0.05):

    data_shape = input_signal.shape  # input_signal has to be a numpy array.
    assert noise_type in ['bsc', 'awgn', 'fading', 'radar', 't-dist', 'isi_perfect', 'isi_uncertain'], "Invalid noise type"
    device = input_signal.device
    if noise_type == 'awgn':
        dist = Normal(torch.tensor([0.0], device = device), torch.tensor([sigma], device = device))
        noise = dist.sample(input_signal.shape).squeeze()
        corrupted_signal = input_signal + noise

    elif noise_type == 'fading':
        fading_h = torch.sqrt(torch.randn_like(input_signal)**2 +  torch.randn_like(input_signal)**2)/np.sqrt(3.14/2.0)
        noise = sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = fading_h *(input_signal) + noise

    elif noise_type == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], data_shape,
                                       p=[1 - radar_prob, radar_prob])

        corrupted_signal = radar_power* np.random.standard_normal( size = data_shape ) * add_pos
        noise = sigma * torch.randn_like(input_signal) +\
                    torch.from_numpy(corrupted_signal).float().to(input_signal.device)
        corrupted_signal = input_signal + noise

    elif noise_type == 't-dist':
        dist = StudentT(torch.tensor([vv], device = device))
        noise = sigma* dist.sample(input_signal.shape).squeeze()
        corrupted_signal = input_signal + noise

    return corrupted_signal

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def min_sum_log_sum_exp(x, y):

    log_sum_ms = torch.min(torch.abs(x), torch.abs(y))*torch.sign(x)*torch.sign(y)
    return log_sum_ms

def min_sum_log_sum_exp_4(x_1, x_2, x_3, x_4):
    return min_sum_log_sum_exp(min_sum_log_sum_exp(x_1, x_2), min_sum_log_sum_exp(x_3, x_4))

def log_sum_exp(x, y):
    def log_sum_exp_(LLR_vector):

        sum_vector = LLR_vector.sum(dim=1, keepdim=True)
        sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)

        return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1) 


    Lv = log_sum_exp_(torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2).permute(0, 2, 1))
    return Lv 

def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing
    bits (0 and 1).
    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.
    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in range(length-2):
        bitarray[bit_width-i-1] = int(binary_string[length-i-1])

    return bitarray


def countSetBits(n):

    count = 0
    while (n):
        n &= (n-1)
        count+= 1

    return count

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, enc_quantize_level = 2, enc_value_limit = 1.0, enc_grad_limit = 0.01, enc_clipping = 'both'):

        ctx.save_for_backward(inputs)
        assert enc_clipping in ['both', 'inputs']
        ctx.enc_clipping = enc_clipping
        ctx.enc_value_limit = enc_value_limit
        ctx.enc_quantize_level = enc_quantize_level
        ctx.enc_grad_limit = enc_grad_limit

        x_lim_abs  = enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.enc_value_limit]=0
            grad_output[input<-ctx.enc_value_limit]=0

        if ctx.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.enc_grad_limit, ctx.enc_grad_limit)
        grad_input = grad_output.clone()

        return grad_input, None


def pairwise_distances(codebook):
    dists = []
    for row1, row2 in combinations(codebook, 2): 
        distance = (row1-row2).pow(2).sum()
        dists.append(np.sqrt(distance.item()))
    return dists, np.min(dists)