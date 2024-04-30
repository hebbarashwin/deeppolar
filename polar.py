import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import sys
import time
from collections import namedtuple

from utils import snr_db2sigma, min_sum_log_sum_exp, log_sum_exp, errors_ber, errors_bler, corrupt_signal, countSetBits


def get_args():
    parser = argparse.ArgumentParser(description='(N,K) Polar code')

    parser.add_argument('--N', type=int, default=4, help='Polar code parameter N')
    parser.add_argument('--K', type=int, default=3, help='Polar code parameter K')
    parser.add_argument('--rate_profile', type=str, default='polar', choices=['RM', 'polar', 'sorted', 'sorted_last', 'rev_polar'], help='Polar rate profiling')
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true')
    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true')
    parser.add_argument('--list_size', type=int, default=1, help='SC List size')
    parser.add_argument('--crc_len', type=int, default='0', choices=[0, 3, 8, 16], help='CRC length')

    parser.add_argument('--batch_size', type=int, default=10000, help='size of the batches')
    parser.add_argument('--test_ratio', type = float, default = 1, help = 'Number of test samples x batch_size')
    parser.add_argument('--test_snr_start', type=float, default=-2., help='testing snr start')
    parser.add_argument('--test_snr_end', type=float, default=4., help='testing snr end')
    parser.add_argument('--snr_points', type=int, default=7, help='testing snr num points')
    args = parser.parse_args()

    return args

class PolarCode:

    def __init__(self, n, K, Fr = None, rs = None, use_cuda = True, infty = 1000., hard_decision = False, lse = 'lse'):

        assert n>=1
        self.n = n
        self.N = 2**n
        self.K = K
        self.G2 = np.array([[1,1],[0,1]])
        self.G = np.array([1])
        for i in range(n):
            self.G = np.kron(self.G, self.G2)
        self.G = torch.from_numpy(self.G).float()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.infty = infty
        self.hard_decision = hard_decision
        self.lse = lse

        if Fr is not None:
            assert len(Fr) == self.N - self.K
            self.frozen_positions = Fr
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()

            self.info_positions = np.array(list(set(self.frozen_positions) ^ set(np.arange(self.N))))
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
            
        else:
            if rs is None:
                # in increasing order of reliability
                self.reliability_seq = np.arange(1023, -1, -1)
                self.rs = self.reliability_seq[self.reliability_seq<self.N]
            else:
                self.reliability_seq = rs
                self.rs = self.reliability_seq[self.reliability_seq<self.N]

                assert len(self.rs) == self.N
            # best K bits
            self.info_positions = self.rs[:self.K]
            self.unsorted_info_positions = self.reliability_seq[self.reliability_seq<self.N][:self.K]
            self.info_positions.sort()
            self.unsorted_info_positions=np.flip(self.unsorted_info_positions)
            # worst N-K bits
            self.frozen_positions = self.rs[self.K:]
            self.unsorted_frozen_positions = self.rs[self.K:]
            self.frozen_positions.sort()


            self.CRC_polynomials = {
            3: torch.Tensor([1, 0, 1, 1]).int(),
            8: torch.Tensor([1, 1, 1, 0, 1, 0, 1, 0, 1]).int(),
            16: torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).int(),
                                    }

    def get_G(self, ell):
        n = int(np.log2(ell))
        G = np.array([1])
        for i in range(n):
            G = np.kron(G, self.G2)
        return G

    def encode_plotkin(self, message, scaling = None, custom_info_positions = None):

        # message shape is (batch, k)
        # BPSK convention : 0 -> +1, 1 -> -1
        # Therefore, xor(a, b) = a*b
        if custom_info_positions is not None:
            info_positions = custom_info_positions
        else:
            info_positions = self.info_positions
        u = torch.ones(message.shape[0], self.N, dtype=torch.float).to(message.device)
        u[:, info_positions] = message

        for d in range(0, self.n):
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
                # u[:, i:i+num_bits] = u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits].clone
        if scaling is not None:
            u = (scaling * np.sqrt(self.N)*u)/torch.norm(scaling)
        return u
    
    def channel(self, code, snr, noise_type = 'awgn', vv =5.0, radar_power = 20.0, radar_prob = 5e-2):
        if noise_type != "bsc":
            sigma = snr_db2sigma(snr)
        else:
            sigma = snr

        r = corrupt_signal(code, sigma, noise_type, vv, radar_power, radar_prob)

        return r

    def define_partial_arrays(self, llrs):
        # Initialize arrays to store llrs and partial_sums useful to compute the partial successive cancellation process.
        llr_array = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        llr_array[:, self.n] = llrs
        partial_sums = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        return llr_array, partial_sums


    def updateLLR(self, leaf_position, llrs, partial_llrs = None, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(self.N) #priors
        llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth, 0, leaf_position, prior, decoded_bits)
        return llrs, decoded_bits


    def partial_decode(self, llrs, partial_llrs, depth, bit_position, leaf_position, prior, decoded_bits=None):
        # Function to call recursively, for partial SC decoder.
        # We are assuming that u_0, u_1, .... , u_{leaf_position -1} bits are known.
        # Partial sums computes the sums got through Plotkin encoding operations of known bits, to avoid recomputation.
        # this function is implemented for rate 1 (not accounting for frozen bits in polar SC decoding)

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (depth - 1)
        leaf_position_at_depth = leaf_position // 2**(depth-1) # will tell us whether left_child or right_child

        # n = 2 tree case
        if depth == 1:
            # Left child
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                u_hat = partial_llrs[:, depth-1, left_bit_position:left_bit_position+1]
            elif leaf_position_at_depth == left_bit_position:
                if self.lse == 'minsum':
                    Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                elif self.lse == 'lse':
                    Lu = log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                #print(Lu.device, prior.device, torch.ones_like(Lu).device)
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu + prior[left_bit_position]*torch.ones_like(Lu)
                if self.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv + prior[right_bit_position] * torch.ones_like(Lv)
                if self.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                u_hat = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
            else:
                if self.lse == 'minsum':
                    Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                elif self.lse == 'lse':
                    # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                    Lu = log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])

                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1

            Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums(self, leaf_position, decoded_bits, partial_llrs):

        u = decoded_bits.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        partial_llrs[:, self.n] = u
        return partial_llrs

    def sc_decode_new(self, corrupted_codewords, snr, use_gt = None, channel = 'awgn'):

        assert channel in ['awgn', 'bsc']

        if channel == 'awgn':
            noise_sigma = snr_db2sigma(snr)
            llrs = (2/noise_sigma**2)*corrupted_codewords
        elif channel == 'bsc':
            # snr refers to transition prob
            p = (torch.ones(1)*(snr + 1e-9)).to(corrupted_codewords.device)
            llrs = (torch.clip(torch.log((1 - p) / p), -10000, 10000) * (corrupted_codewords + 1) - torch.clip(torch.log(p / (1-p)), -10000, 10000) * (corrupted_codewords - 1))/2

        # step-wise implementation using updateLLR and updatePartialSums

        priors = torch.zeros(self.N)
        priors[self.frozen_positions] = self.infty

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            #start = time.time()
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_llrs, priors)
            #print('SC update : {}'.format(time.time() - start), corrupted_codewords.shape[0])
            if use_gt is None:
                u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
            else:
                u_hat[:, ii] = use_gt[:, ii]
            #start = time.time()
            partial_llrs = self.updatePartialSums(ii, u_hat, partial_llrs)
            #print('SC partial: {}s, {}', time.time() - start, 'frozen' if ii in self.frozen_positions else 'info')
        decoded_bits = u_hat[:, self.info_positions]
        return llr_array[:, 0, :].clone(), decoded_bits

    def get_CRC(self, message):

        # need to optimize.
        # inout message should be int

        padded_bits = torch.cat([message, torch.zeros(self.CRC_len).int().to(message.device)])
        while len(padded_bits[0:self.K_minus_CRC].nonzero()):
            cur_shift = (padded_bits != 0).int().argmax(0)
            padded_bits[cur_shift: cur_shift + self.CRC_len + 1] = padded_bits[cur_shift: cur_shift + self.CRC_len + 1] ^ self.CRC_polynomials[self.CRC_len].to(message.device)

        return padded_bits[self.K_minus_CRC:]

    def CRC_check(self, message):

        # need to optimize.
        # input message should be int

        padded_bits = message
        while len(padded_bits[0:self.K_minus_CRC].nonzero()):
            cur_shift = (padded_bits != 0).int().argmax(0)
            padded_bits[cur_shift: cur_shift + polar.CRC_len + 1] ^= self.CRC_polynomials[self.CRC_len].to(message.device)

        if padded_bits[self.K_minus_CRC:].sum()>0:
            return 0
        else:
            return 1

    def encode_with_crc(self, message, CRC_len):
        self.CRC_len = CRC_len
        self.K_minus_CRC = self.K - CRC_len

        if CRC_len == 0:
            return self.encode_plotkin(message)
        else:
            crcs = 1-2*torch.vstack([self.get_CRC((0.5+0.5*message[jj]).int()) for jj in range(message.shape[0])])
            encoded = self.encode_plotkin(torch.cat([message, crcs], 1))

            return encoded

def get_frozen(N, K, rate_profile, target_K = None):
    n = int(np.log2(N))
    if rate_profile == 'polar':
        # computed for SNR = 0
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])

            # for RM :(
            # rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 3, 5, 8, 4, 2, 1, 0])

        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])
        elif n<9:
            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
        else:
            rs = np.array([1023, 1022, 1021, 1019, 1015, 1007, 1020,  991, 1018, 1017, 1014,
       1006,  895, 1013, 1011,  959, 1005,  990, 1003,  989,  767, 1016,
        999, 1012,  987,  958,  983,  957, 1010, 1004,  955, 1009,  894,
        975,  893, 1002,  951, 1001,  988,  511,  766,  998,  891,  943,
        986,  997,  985,  887,  956,  765,  995,  927,  982,  981,  879,
        954,  974,  763,  953,  979,  510, 1008,  759,  863,  950,  892,
       1000,  973,  949,  509,  890,  971,  996,  942,  751,  984,  889,
        507,  947,  831,  886,  967,  941,  764,  926,  980,  994,  939,
        885,  993,  735,  878,  925,  503,  762,  883,  978,  935,  703,
        495,  952,  877,  761,  972,  923,  977,  948,  758,  862,  875,
        919,  970,  757,  861,  508,  969,  750,  946,  479,  888,  639,
        871,  911,  830,  940,  859,  755,  966,  945,  749,  506,  884,
        938,  965,  829,  734,  924,  855,  505,  747,  963,  937,  882,
        934,  827,  733,  447,  992,  847,  876,  501,  921,  702,  494,
        881,  760,  743,  933,  502,  918,  874,  922,  823,  731,  499,
        860,  756,  931,  701,  873,  493,  727,  917,  870,  976,  815,
        910,  383,  968,  478,  858,  754,  699,  491,  869,  944,  748,
        638,  915,  477,  719,  909,  964,  255,  799,  504,  857,  854,
        753,  828,  746,  695,  487,  907,  637,  867,  853,  475,  936,
        962,  446,  732,  826,  745,  846,  500,  825,  903,  687,  932,
        635,  471,  445,  742,  880,  498,  730,  851,  822,  382,  920,
        845,  741,  443,  700,  729,  631,  492,  872,  961,  726,  821,
        930,  497,  381,  843,  463,  916,  739,  671,  623,  490,  929,
        439,  814,  819,  868,  752,  914,  698,  725,  839,  856,  476,
        813,  718,  908,  486,  723,  866,  489,  607,  431,  697,  379,
        811,  798,  913,  575,  717,  254,  694,  636,  474,  807,  715,
        906,  797,  693,  865,  960,  852,  744,  634,  473,  795,  905,
        485,  415,  483,  470,  444,  375,  850,  740,  686,  902,  824,
        691,  253,  711,  633,  844,  685,  630,  901,  367,  791,  928,
        728,  820,  849,  783,  670,  899,  738,  842,  683,  247,  469,
        441,  442,  462,  251,  737,  438,  467,  351,  629,  841,  724,
        679,  669,  496,  461,  818,  380,  437,  627,  622,  459,  378,
        239,  488,  667,  838,  430,  484,  812,  621,  319,  817,  435,
        377,  696,  722,  912,  606,  810,  864,  716,  837,  721,  714,
        809,  796,  455,  472,  619,  835,  692,  663,  223,  414,  904,
        427,  806,  482,  632,  713,  690,  848,  605,  373,  252,  794,
        429,  710,  684,  615,  805,  900,  655,  468,  366,  603,  413,
        574,  481,  371,  250,  793,  466,  423,  374,  689,  628,  440,
        365,  709,  789,  803,  411,  573,  682,  249,  460,  790,  668,
        599,  350,  707,  246,  681,  465,  571,  626,  436,  407,  782,
        191,  127,  363,  620,  666,  458,  245,  349,  677,  434,  678,
        591,  787,  399,  457,  359,  238,  625,  840,  567,  736,  665,
        428,  376,  781,  898,  618,  675,  318,  454,  662,  243,  897,
        347,  836,  816,  720,  433,  604,  617,  779,  808,  661,  834,
        712,  804,  833,  559,  237,  453,  426,  222,  317,  775,  372,
        343,  412,  235,  543,  614,  451,  425,  422,  613,  370,  221,
        315,  480,  335,  659,  654,  364,  190,  369,  248,  653,  688,
        231,  410,  602,  611,  802,  792,  421,  651,  601,  598,  708,
        311,  219,  572,  597,  788,  570,  409,  590,  362,  801,  680,
        464,  406,  419,  348,  647,  786,  215,  589,  706,  361,  676,
        566,  189,  595,  244,  569,  303,  405,  358,  456,  346,  398,
        565,  242,  126,  705,  780,  587,  624,  664,  236,  187,  357,
        432,  785,  558,  674,  207,  403,  397,  452,  345,  563,  778,
        241,  316,  342,  616,  660,  557,  125,  234,  183,  287,  355,
        583,  673,  395,  424,  314,  220,  777,  341,  612,  658,  123,
        175,  774,  555,  233,  334,  542,  450,  313,  391,  230,  652,
        368,  218,  339,  600,  119,  333,  657,  610,  773,  541,  310,
        420,  159,  229,  650,  551,  596,  609,  408,  217,  449,  188,
        309,  214,  331,  111,  539,  360,  771,  649,  302,  418,  594,
        896,  227,  404,  646,  186,  588,  832,  568,  213,  417,  301,
        307,  356,  402,  800,  564,  327,   95,  206,  240,  535,  593,
        645,  586,  344,  396,  185,  401,  211,  354,  299,  585,  286,
        562,  643,  182,  205,  124,  232,  285,  295,  181,  556,  582,
        527,  394,  340,   63,  203,  561,  353,  448,  122,  283,  393,
        581,  554,  174,  390,  704,  312,  338,  228,  179,  784,  199,
        553,  121,  173,  389,  540,  579,  332,  118,  672,  550,  337,
        158,  279,  271,  416,  216,  308,  387,  538,  549,  226,  330,
        776,  171,  212,  117,  110,  329,  656,  157,  772,  306,  326,
        225,  167,  115,  537,  534,  184,  109,  300,  547,  305,  210,
        155,  533,  325,  352,  608,  400,  298,  204,   94,  648,  284,
        209,  151,  180,  107,  770,  297,  392,  323,  592,  202,  644,
         93,  294,  178,  103,  143,  282,   62,  336,  201,  120,  172,
        198,  769,  584,   91,  388,  293,  177,  526,  278,  281,  642,
        525,  531,   61,  170,  116,  197,   87,  156,  277,  114,  560,
        169,   59,  291,  580,  275,  523,  641,  270,  195,  552,  519,
        166,  224,  578,  108,  269,   79,  154,  113,  548,  577,  536,
        328,   55,  106,  165,  153,  150,  386,  208,  324,  546,  385,
        267,   47,   92,  163,  296,  304,  105,  102,  149,  263,  532,
        322,  292,  545,   90,  200,   31,  321,  530,  142,  176,  147,
        101,  141,  196,  524,  529,  290,   89,  280,   60,   86,   99,
        139,  168,   58,  522,  276,   85,  194,  289,   78,  135,  112,
        521,   57,   83,   54,  518,  274,  268,  768,  164,   77,  152,
        193,   53,  162,  104,  517,  273,  266,   75,   46,  148,   51,
        640,  100,   45,  576,  161,  265,  262,   71,  146,   30,  140,
         88,  515,   98,   43,   29,  261,  145,  138,   84,  259,   39,
         97,   27,   56,   82,  137,   76,  384,  134,   23,   52,  133,
        320,   15,   73,   50,   81,  131,   44,   70,  544,  192,  528,
        288,  520,  160,  272,   74,   49,  516,   42,   69,   28,  144,
         41,   67,   96,  514,   38,  264,  260,  136,   22,   25,   37,
         80,  513,   26,  258,   35,  132,   21,  257,   72,   14,   48,
         13,   19,  130,   68,   40,   11,  512,   66,  129,    7,   36,
         24,   34,  256,   20,   65,   33,   12,  128,   18,   10,   17,
          6,    9,   64,    5,    3,   32,   16,    8,    4,    2,    1,
          0])
        rs = rs[rs<N]
        Fr = rs[K:].copy()
        Fr.sort()

    elif rate_profile == 'RM':
        rmweight = np.array([countSetBits(i) for i in range(N)])
        Fr = np.argsort(rmweight)[:-K]
        Fr.sort()

    elif rate_profile == 'sorted':
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<N]
        first_inds = rs[:K].copy()
        first_inds.sort()
        rs[:K] = first_inds

        Fr = rs[K:].copy()
        Fr.sort()

    elif rate_profile == 'sorted_last':
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<N]
        first_inds = rs[:K].copy()
        first_inds.sort()
        rs[:K] = first_inds[::-1]

        Fr = rs[K:].copy()
        Fr.sort()

    elif rate_profile == 'rev_polar':

        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<N]
        first_inds = rs[:target_K].copy()
        rs[:target_K] = first_inds[::-1]
        Fr = rs[K:].copy()
        Fr.sort()

    return Fr

if __name__ == '__main__':
    args = get_args()

    n = int(np.log2(args.N))


    Fr = get_frozen(args.N, args.K, args.rate_profile)
    polar = PolarCode(n, args.K, Fr = Fr, hard_decision=True)

    # Multiple SNRs:
    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start)* 1.0 /  (args.snr_points-1)
        snr_range = [snrs_interval* item + args.test_snr_start for item in range(args.snr_points)]

    if args.only_args:
        print("Loaded args. Exiting")
        sys.exit()

    bers_SC = [0. for ii in snr_range]
    blers_SC = [0. for ii in snr_range]

    for r in range(int(args.test_ratio)):
        msg_bits = 1 - 2*(torch.rand(args.batch_size, args.K) > 0.5).float()
        codes = polar.encode_plotkin(msg_bits)


        for snr_ind, snr in enumerate(snr_range):

            # codes_G = polar.encode_G(msg_bits_bpsk)
            noisy_code = polar.channel(codes, snr)
            noise = noisy_code - codes

            SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
            ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign()).item()
            bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign()).item()

            bers_SC[snr_ind] += ber_SC/args.test_ratio
            blers_SC[snr_ind] += bler_SC/args.test_ratio

    print("Test SNRs : ", snr_range)
    print("BERs of SC: {0}".format(bers_SC))
    print("BLERs of SC: {0}".format(blers_SC))
