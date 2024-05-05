import torch 
import torch.nn as nn 
import torch.nn.functional as F


import numpy as np
from polar import PolarCode, get_frozen
from models import g_Full, f_Full, weights_init
from utils import min_sum_log_sum_exp, min_sum_log_sum_exp_4, countSetBits, log_sum_exp, STEQuantize
from collections import defaultdict
import os

class DeepPolar(PolarCode):
    def __init__(self, args, device, N, K, ell = 2, infty = 1000., depth_map : defaultdict = None):

        # rmweight = np.array([countSetBits(i) for i in range(N)])
        # Frozen = np.argsort(rmweight)[:-K]
        # Frozen.sort()

        self.args = args
        Fr = get_frozen(N, K, self.args.rate_profile)
        super().__init__(n = int(np.log2(N)), K = K, Fr=Fr,  infty = infty)
        self.N = N

        if depth_map is not None:
            # depth map is a dict, product of values should be equal to N
            assert np.prod(list(depth_map.values())) == N
            # assert that keys od depth map start from one and go continuosly till some point 
            assert min(list(depth_map.keys())) == 1
            assert max(list(depth_map.keys())) <= int(np.log2(N))
            self.ell = None
            self.n_ell = len(depth_map.keys())
            assert max(list(depth_map.keys())) == self.n_ell

            self.depth_map = depth_map
        else:
            self.ell = ell
            self.n_ell = int(np.log(N)/np.log(self.ell))

            self.depth_map = defaultdict(int)
            for d in range(1, self.n_ell+1):
                self.depth_map[d] = self.ell
            assert np.prod(list(self.depth_map.values())) == N

        self.device = device
        self.fnet_dict = None
        self.gnet_dict = None

        self.infty = infty

    @staticmethod
    def get_onehot(actions):
        inds = (0.5 + 0.5*actions).long()
        if len(actions.shape) == 2:
            return torch.eye(2, device = inds.device)[inds].reshape(actions.shape[0], -1)
        elif len(actions.shape) == 3:
            return torch.eye(2, device = inds.device)[inds].reshape(actions.shape[0], actions.shape[1], -1)

    def define_kernel_nns(self, ell, unfrozen = None, fnet = 'KO', gnet = 'KO', shared = False):

        if 'KO' in fnet:
            self.fnet_dict = {}
        else:
            self.fnet_dict = None

        self.shared = shared
        if 'KO' in gnet:
            self.gnet_dict = {}
        else:
            self.gnet_dict = None
        dec_hidden_size = self.args.dec_hidden_size
        enc_hidden_size = self.args.enc_hidden_size

        depth = 1
        assert len(unfrozen) > 0, "No unfrozen bits!"

        self.fnet_dict[depth] = {}

        if fnet == 'KO_parallel' or fnet == 'KO_last_parallel':
            bit_position = 0
                   
            self.fnet_dict[depth][bit_position] = {}
            # input_size = self.N if depth == self.n_ell else self.N // int(np.prod([self.depth_map[d] for d in range(depth+1, self.n_ell+1)]))
            input_size = ell             
            # For curriculum, only for lowest depth.
            output_size = ell#len(unfrozen)
            self.fnet_dict[depth][bit_position] = f_Full(input_size, dec_hidden_size, output_size, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
 
        elif 'KO' in fnet:
            if shared:
                self.fnet_dict[depth] = {}
                for current_position in range(ell):
                    self.fnet_dict[depth][current_position] = f_Full(ell + current_position, dec_hidden_size, 1, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
            else:
                bit_position = 0
                for current_position in unfrozen:
                    if not self.fnet_dict[depth].get(bit_position):
                        self.fnet_dict[depth][bit_position] = {}
                    input_size = ell + (int(self.args.onehot)+1)*current_position
                    self.fnet_dict[depth][bit_position][current_position] = f_Full(input_size, dec_hidden_size, 1, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
                
        if 'KO' in gnet:
            self.gnet_dict[depth] = {}
            if shared:
                if gnet == 'KO':
                    self.gnet_dict[depth] = g_Full(ell, enc_hidden_size, ell-1, depth = self.args.g_depth, skip_depth = self.args.g_skip_depth, skip_layer = self.args.g_skip_layer, ell = ell, activation = self.args.enc_activation, use_skip = self.args.skip).to(self.device)
            else:
                bit_position = 0
                if gnet == 'KO':
                    self.gnet_dict[depth][bit_position] = g_Full(ell, enc_hidden_size, ell-1, depth = self.args.g_depth, skip_depth = self.args.g_skip_depth, skip_layer = self.args.g_skip_layer, ell = ell, activation = self.args.enc_activation, use_skip = self.args.skip).to(self.device)

    def define_and_load_nns(self, ell, kernel_load_path = None, fnet = 'KO', gnet = 'KO', shared = True, dataparallel = False):

        if 'KO' in fnet:
            self.fnet_dict = {}
        else:
            self.fnet_dict = None

        self.shared = shared
        if 'KO' in gnet:
            self.gnet_dict = {}
        else:
            self.gnet_dict = None
        dec_hidden_size = self.args.dec_hidden_size
        enc_hidden_size = self.args.enc_hidden_size

        for depth in range(self.n_ell, 0, -1):
            if depth in self.args.polar_depths:
                continue
            ell = self.depth_map[depth]
            proj_size = np.prod([self.depth_map[d] for d in range(1, depth+1)])

            if fnet == 'KO_last_parallel' and depth == 1:
                self.fnet_dict[depth] = {}
                for bit_position in range(self.N // proj_size):
                    proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
                    get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
                    num_info_in_proj = get_num_info_proj(proj)

                    subproj_len = len(proj) // ell
                    subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
                    num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
                    unfrozen = [i for i, x in enumerate(num_info_in_subproj) if x >= 1]
                    
                    # input_size = self.N if depth == self.n_ell else self.N // int(np.prod([self.depth_map[d] for d in range(depth+1, self.n_ell+1)]))
                    input_size = ell             
                    # For curriculum, only for lowest depth.
                    output_size = ell
                    self.fnet_dict[depth][bit_position] = f_Full(input_size, dec_hidden_size, output_size, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
                    if len(unfrozen) > 0:
                        if kernel_load_path is not None:
                            try:
                                ckpt = torch.load(os.path.join(kernel_load_path + '_parallel', f'{ell}_{len(unfrozen)}.pt'))
                                ckpt_exists = True
                            except FileNotFoundError:
                                print(f"Parallel File not found for ell = {ell}, num_unfrozen = {len(unfrozen)}")
                                ckpt_exists = False
                                pass 
                        else:
                            ckpt_exists = False
                        if ckpt_exists :
                            f_ckpt = ckpt[0][1][0].state_dict()
                            self.fnet_dict[depth][bit_position].load_state_dict(f_ckpt)
                        if dataparallel:
                            self.fnet_dict[depth][bit_position] = nn.Dataparallel(self.fnet_dict[depth][bit_position])
   
            elif 'KO' in fnet:
                self.fnet_dict[depth] = {}
                if shared:
                    self.fnet_dict[depth] = {}
                    for current_position in range(ell):
                        self.fnet_dict[depth][current_position] = f_Full(ell + current_position, dec_hidden_size, 1, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
                        if dataparallel:
                            self.fnet_dict[depth][current_position] = nn.DataParallel(self.fnet_dict[depth][current_position])
                else:

                    for bit_position in range(self.N // proj_size):
                        proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
                        get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
                        num_info_in_proj = get_num_info_proj(proj)

                        subproj_len = len(proj) // ell
                        subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
                        num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
                        unfrozen = [i for i, x in enumerate(num_info_in_subproj) if x >= 1]
                        if len(unfrozen) > 0:
                            if kernel_load_path is not None:
                                try:
                                    ckpt = torch.load(os.path.join(kernel_load_path, f'{ell}_{len(unfrozen)}.pt'))
                                    ckpt_exists = True
                                except FileNotFoundError:
                                    print(f"File not found for ell = {ell}, num_unfrozen = {len(unfrozen)}")
                                    ckpt_exists = False
                                    pass 
                            else:
                                ckpt_exists = False
                        for current_position in unfrozen:
                            if not self.fnet_dict[depth].get(bit_position):
                                self.fnet_dict[depth][bit_position] = {}                          
                            input_size = ell + (int(self.args.onehot)+1)*current_position
                            output_size = 1

                            # if current_position == 0:
                            #     self.fnet_dict[depth][bit_position][current_position] = f_Full(ell**depth, dec_hidden_size, 1, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth).to(self.device)
                            # else:
                            self.fnet_dict[depth][bit_position][current_position] = f_Full(input_size, dec_hidden_size, output_size, activation = self.args.dec_activation, dropout_p = self.args.dropout_p, depth = self.args.f_depth, use_norm = self.args.use_norm).to(self.device)
                            if ckpt_exists :
                                try:
                                    f_ckpt = ckpt[0][1][0][current_position].state_dict()
                                except:
                                    print(unfrozen)
                                self.fnet_dict[depth][bit_position][current_position].load_state_dict(f_ckpt)
                            if dataparallel:
                                self.fnet_dict[depth][bit_position][current_position] = nn.DataParallel(self.fnet_dict[depth][bit_position][current_position])

            if 'KO' in gnet:
                self.gnet_dict[depth] = {}
                if shared:
                    if gnet == 'KO':
                        if not dataparallel:
                            self.gnet_dict[depth] = g_Full(ell, enc_hidden_size, ell-1, depth = self.args.g_depth, skip_depth = self.args.g_skip_depth, skip_layer = self.args.g_skip_layer, ell = ell, use_skip = self.args.skip).to(self.device)
                        else:
                            self.gnet_dict[depth] = nn.DataParallel(g_Full(ell, enc_hidden_size, ell-1, depth = self.args.g_depth, skip_depth = self.args.g_skip_depth, skip_layer = self.args.g_skip_layer, ell = ell, use_skip = self.args.skip)).to(self.device)
                else:
                    for bit_position in range(self.N // proj_size):
                        proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
                        get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
                        num_info_in_proj = get_num_info_proj(proj)

                        subproj_len = len(proj) // ell
                        subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
                        num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
                        unfrozen = [i for i, x in enumerate(num_info_in_subproj) if x >= 1]

                        if num_info_in_proj > 0:
                            if gnet == 'KO':
                                self.gnet_dict[depth][bit_position] = g_Full(ell, enc_hidden_size, ell-1, depth = self.args.g_depth, skip_depth = self.args.g_skip_depth, skip_layer = self.args.g_skip_layer, ell = ell, activation = self.args.enc_activation, use_skip = self.args.skip).to(self.device)
                            if kernel_load_path is not None:
                                try:
                                    ckpt = torch.load(os.path.join(kernel_load_path, f'{ell}_{len(unfrozen)}.pt'))
                                    self.gnet_dict[depth][bit_position].load_state_dict(ckpt[1][1][0].state_dict())
                                except FileNotFoundError:
                                    print(f"File not found for ell = {ell}, num_unfrozen = {len(unfrozen)}")
                                    pass
                            if dataparallel:
                                self.gnet_dict[depth][bit_position] = nn.DataParallel(self.gnet_dict[depth][bit_position])


                        # print(f"g : {depth}, {bit_position}, {len(unfrozen)}")
        if kernel_load_path is not None:
            print("Loaded kernel from ", kernel_load_path)

    def load_nns(self, fnet_dict, gnet_dict = None, shared = False):
        self.fnet_dict = fnet_dict
        self.gnet_dict = gnet_dict

        for depth in fnet_dict.keys():
            if self.fnet_dict is not None:
                for bit_position in self.fnet_dict[depth].keys():
                    if not isinstance(self.fnet_dict[depth][bit_position], dict):#shared or self.args.decoder_type == 'KO_parallel' or self.args.decoder_type == 'KO_RNN':
                        self.fnet_dict[depth][bit_position].to(self.device)
                    else:
                        for current_position in self.fnet_dict[depth][bit_position].keys():
                            self.fnet_dict[depth][bit_position][current_position].to(self.device)
            if gnet_dict is not None:
                if shared:
                    self.gnet_dict[depth].to(self.device)
                else:
                    for bit_position in self.gnet_dict[depth].keys():
                        self.gnet_dict[depth][bit_position].to(self.device)
        print("NN weights loaded!")

    def load_partial_nns(self, fnet_dict, gnet_dict = None):

        for depth in fnet_dict.keys():
            if fnet_dict is not None:
                for bit_position in fnet_dict[depth].keys():
                    if isinstance(fnet_dict[depth][bit_position], dict):
                        for current_position in fnet_dict[depth][bit_position].keys():
                            self.fnet_dict[depth][bit_position][current_position] = fnet_dict[depth][bit_position][current_position].to(self.device)
                    else:
                        self.fnet_dict[depth][bit_position] = fnet_dict[depth][bit_position].to(self.device)

            if gnet_dict is not None:
                for bit_position in gnet_dict[depth].keys():
                    self.gnet_dict[depth][bit_position] = gnet_dict[depth][bit_position].to(self.device)
        print("NN weights loaded!")

    def kernel_encode(self, ell, gnet, msg_bits, info_positions, binary = False):
        input_shape = msg_bits.shape[-1]
        assert input_shape <= ell
        u = torch.ones(msg_bits.shape[0], self.N, dtype=torch.float).to(self.device)
        u[:, info_positions] = msg_bits
        output =torch.cat([gnet(u.unsqueeze(1)).squeeze(1), u[:, -1:]], 1)

        power_constrained_u = self.power_constraint(output)
        if binary:
            stequantize = STEQuantize.apply
            power_constrained_u = stequantize(power_constrained_u)
        return power_constrained_u

    def deeppolar_encode(self, msg_bits, binary = False):
        u = torch.ones(msg_bits.shape[0], self.N, dtype=torch.float).to(self.device)
        u[:, self.info_positions] = msg_bits
        for d in range(1, self.n_ell+1):
            # num_bits = self.ell**(d-1)
            num_bits = np.prod([self.depth_map[dd] for dd in range(1, d)]) if d > 1 else 1
            # proj_size = self.ell**(d)
            proj_size = np.prod([self.depth_map[dd] for dd in range(1, d+1)])
            ell = self.depth_map[d]
            for bit_position, i in enumerate(np.arange(0, self.N, ell*num_bits)):

                # [u v] encoded to [(u xor v),v)]
                proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
                get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
                num_info_in_proj = get_num_info_proj(proj)

                subproj_len = len(proj) // ell
                subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
                num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
                num_nonzero_subproj = sum([int(x != 0) for x in num_info_in_subproj])
                
                if num_info_in_proj > 0:
                    info_bits_present = True          
                else:
                    info_bits_present = False         
                if d in self.args.polar_depths:
                    info_bits_present = False

                enc_chunks = []
                ell = self.depth_map[d]
                for j in range(ell):
                    chunk = u[:, i + j*num_bits:i + (j+1)*num_bits].unsqueeze(2).clone()
                    enc_chunks.append(chunk)
                if info_bits_present:
                    concatenated_chunks = torch.cat(enc_chunks, 2)
                    if self.shared:
                        output = torch.cat([self.gnet_dict[d](concatenated_chunks), u[:, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                    else:
                        output = torch.cat([self.gnet_dict[d][bit_position](concatenated_chunks), u[:, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                    output = output.permute(0,2,1).reshape(msg_bits.shape[0], -1, 1).squeeze(2)

                else:
                    output = self.encode_chunks_plotkin(enc_chunks, ell)
                u = torch.cat((u[:, :i], output, u[:, i + ell*num_bits:]), dim=1)

        power_constrained_u = self.power_constraint(u)
        if binary:
            stequantize = STEQuantize.apply
            power_constrained_u = stequantize(power_constrained_u)
        return power_constrained_u

    def power_constraint(self, codewords):
        return F.normalize(codewords, p=2, dim=1)*np.sqrt(self.N)

    def encode_chunks_plotkin(self, enc_chunks, ell = None):

        # message shape is (batch, k)
        # BPSK convention : 0 -> +1, 1 -> -1
        # Therefore, xor(a, b) = a*b

        # to change for other kernels

        if ell is None:
            ell = self.ell
        assert len(enc_chunks) == ell
        chunk_size = enc_chunks[0].shape[1]
        batch_size = enc_chunks[0].shape[0]

        u = torch.cat(enc_chunks, 1).squeeze(2)
        n = int(np.log2(ell))

        for d in range(0, n):
            num_bits = 2**d * chunk_size
            for i in np.arange(0, chunk_size*ell, 2*num_bits):
                # [u v] encoded to [(u,v) xor v]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        return u
            
    def deeppolar_parallel_decode(self, noisy_code):
        # Successive cancellation decoder for polar codes
        assert noisy_code.shape[1] == self.N

        depth = self.n_ell

        decoded_llrs = self.infty*torch.ones(noisy_code.shape[0], self.N, device = noisy_code.device)
        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)
        decoded_llrs  = self.KO_parallel_decode_depth(noisy_code.unsqueeze(2), depth, 0, decoded_llrs)
        decoded_llrs = decoded_llrs[:, self.info_positions]
        return decoded_llrs, torch.sign(decoded_llrs)

    def deeppolar_parallel_decode_depth(self, llrs, depth, bit_position, decoded_llrs):
        # Function to call recursively, for SC decoder

        # half_index = self.ell ** (depth - 1)
        half_index = np.prod([self.depth_map[d] for d in range(1, depth)]) if depth > 1 else 1
        ell = self.depth_map[depth]
        left_bit_position = self.depth_map[depth] *  bit_position 

        # Check if >1 information bits are present in the current projection. If not, don't use NNs - use polar encoding and minsum SC decoding.
        # proj_size = self.ell**(depth)
        proj_size = np.prod([self.depth_map[d] for d in range(1, depth+1)])

        proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
        get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
        get_info_proj = lambda proj : [x for x in proj if x in self.info_positions]

        num_info_in_proj = get_num_info_proj(proj)
        info_in_proj = get_info_proj(proj)

        subproj_len = len(proj) // ell
        subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
        num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
        num_nonzero_subproj = sum([int(x != 0) for x in num_info_in_subproj])
        unfrozen = np.array([i for i, x in enumerate(num_info_in_subproj) if x >= 1])

        dec_chunks = torch.cat([llrs[:, (j)*half_index:(j+1)*half_index].clone() for j in range(ell)], 2)
        Lu = self.fnet_dict[depth][bit_position](dec_chunks)

        if depth == 1:
            u = torch.tanh(Lu/2)
            decoded_llrs[:, left_bit_position + unfrozen] = Lu.squeeze(1)
        else:
            for index, current_position in enumerate(unfrozen):
                bit_position_offset = left_bit_position + current_position                
                decoded_llrs = self.deeppolar_parallel_decode_depth(Lu[:, :, index:index+1], depth-1, bit_position_offset, decoded_llrs)

        return decoded_llrs
            
    def deeppolar_decode(self, noisy_code):
        assert noisy_code.shape[1] == self.N

        depth = self.n_ell

        decoded_llrs = self.infty*torch.ones(noisy_code.shape[0], self.N, device = noisy_code.device)
        
        # don't want to go into useless frozen subtrees.
        partial_sums = torch.ones(noisy_code.shape[0], self.n_ell+1, self.N, device=noisy_code.device)

        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)

        decoded_llrs, partial_sums = self.deeppolar_decode_depth(noisy_code.unsqueeze(2), depth, 0, decoded_llrs, partial_sums)
        decoded_llrs = decoded_llrs[:, self.info_positions]

        return decoded_llrs, torch.sign(decoded_llrs)
    
    def deeppolar_decode_depth(self, llrs, depth, bit_position, decoded_llrs, partial_sums):
        # Function to call recursively, for SC decoder

        # half_index = self.ell ** (depth - 1)
        half_index = np.prod([self.depth_map[d] for d in range(1, depth)]) if depth > 1 else 1
        ell = self.depth_map[depth]
        left_bit_position = self.depth_map[depth] *  bit_position 

        # Check if >1 information bits are present in the current projection. If not, don't use NNs - use polar encoding and minsum SC decoding.
        # proj_size = self.ell**(depth)
        # size of the projection of tht subtree
        proj_size = np.prod([self.depth_map[d] for d in range(1, depth+1)])

        # This chunk - finds infrozen positions in this kernel.
        proj = np.arange(bit_position*proj_size, (bit_position+1)*proj_size)
        get_num_info_proj = lambda proj : sum([int(x in self.info_positions) for x in proj])
        get_info_proj = lambda proj : [x for x in proj if x in self.info_positions]

        num_info_in_proj = get_num_info_proj(proj)
        info_in_proj = get_info_proj(proj)

        subproj_len = len(proj) // ell
        subproj = [proj[i:i+subproj_len] for i in range(0, len(proj), subproj_len)]
        num_info_in_subproj = [get_num_info_proj(x) for x in subproj]
        num_nonzero_subproj = sum([int(x != 0) for x in num_info_in_subproj])
        unfrozen = np.array([i for i, x in enumerate(num_info_in_subproj) if x >= 1])

        if num_nonzero_subproj > 0:
            info_bits_present = True      
        else:
            info_bits_present = False 

        if depth in self.args.polar_depths:
            info_bits_present = False
                
        # This will be input to decoder
        dec_chunks = [llrs[:, (j)*half_index:(j+1)*half_index].clone() for j in range(ell)]
        # n = 2 tree case
        if depth == 1:
            if self.args.decoder_type == 'KO_last_parallel':
                concatenated_chunks = torch.cat(dec_chunks, 2)
                Lu = self.fnet_dict[depth][bit_position](concatenated_chunks)[:, 0, unfrozen]
                u_hat = torch.tanh(Lu/2)
                decoded_llrs[:, left_bit_position + unfrozen] = Lu
                partial_sums[:, depth-1, left_bit_position + unfrozen] = u_hat

            else:
                for current_position in range(ell):
                    bit_position_offset = left_bit_position + current_position
                    if current_position > 0:
                        # I am adding previously decoded bits . (either onehot or normal)
                        if self.args.onehot:
                            prev_decoded = get_onehot(partial_sums[:, depth-1, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).sign()).detach().clone()
                        else:
                            prev_decoded = partial_sums[:, depth-1, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).clone()
                        dec_chunks.append(prev_decoded)

                    if bit_position_offset in self.frozen_positions: # frozen 
                        # don't update decoded llrs. It already has ones*prior.
                        # actually don't need this. can skip.
                        partial_sums[:, depth-1, bit_position_offset] = torch.ones_like(partial_sums[:, depth-1, bit_position_offset])
                    else: # information bit
                        # This is the decoding.
                        concatenated_chunks = torch.cat(dec_chunks, 2)
                        if self.shared:
                            Lu = self.fnet_dict[depth][current_position](concatenated_chunks)
                        else:
                            Lu = self.fnet_dict[depth][bit_position][current_position](concatenated_chunks)

                        u_hat = torch.tanh(Lu/2).squeeze(2)
                        decoded_llrs[:, bit_position_offset] = Lu.squeeze(2).squeeze(1)
                        partial_sums[:, depth-1, bit_position_offset] = u_hat.squeeze(1)

            # Encoding back the decoded bits - for higher layers.
            # # Compute decoded codeword
            i = left_bit_position * half_index
            # num_bits = self.ell**(depth-1)
            num_bits = 1

            enc_chunks = []
            for j in range(ell):
                chunk = torch.sign(partial_sums[:, depth-1, i + j*num_bits:i + (j+1)*num_bits]).unsqueeze(2).detach().clone()
                enc_chunks.append(chunk)
            if info_bits_present:
                concatenated_chunks = torch.cat(enc_chunks, 2)
                if 'KO' in self.args.encoder_type:
                    if self.shared:
                        output = torch.cat([self.gnet_dict[depth](concatenated_chunks), partial_sums[:, depth-1, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                    else:
                        # bit position of the previous depth.
                        output = torch.cat([self.gnet_dict[depth][bit_position](concatenated_chunks), partial_sums[:, depth-1, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                    output = output.permute(0,2,1).reshape(llrs.shape[0], -1, 1).squeeze(2)
                else:
                    output = self.encode_chunks_plotkin(enc_chunks, ell)
            else:
                output = self.encode_chunks_plotkin(enc_chunks, ell)
            partial_sums[:, depth, i : i + num_bits*ell] = output.clone()
            
            return decoded_llrs, partial_sums

        # General case
        else:
            for current_position in range(ell):
                bit_position_offset = left_bit_position + current_position

                if current_position > 0:
                    if self.args.onehot:
                        prev_decoded = get_onehot(partial_sums[:, depth-1, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).sign()).detach().clone()
                    else:
                        prev_decoded = partial_sums[:, depth-1, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).clone()
                    dec_chunks.append(prev_decoded)
                concatenated_chunks = torch.cat(dec_chunks, 2)

                if current_position in unfrozen:
                    # General decoding ....
                    # add the decoded bit here
                    if self.shared:
                        Lu = self.fnet_dict[depth][current_position](concatenated_chunks).squeeze(2)
                    else:
                        # if current_position == 0:
                        #     Lu = self.fnet_dict[depth][bit_position][current_position](llrs)
                        # else:
                        Lu = self.fnet_dict[depth][bit_position][current_position](concatenated_chunks)
                    decoded_llrs, partial_sums = self.deeppolar_decode_depth(Lu, depth-1, bit_position_offset, decoded_llrs, partial_sums)
                else:
                    Lu = self.infty*torch.ones_like(llrs)


            # Compute decoded codeword
            if depth < self.n_ell :
                i = left_bit_position * half_index
                # num_bits = self.ell**(depth-1)
                num_bits = np.prod([self.depth_map[d] for d in range(1, depth)])
                enc_chunks = []
                for j in range(ell):
                    chunk = torch.sign(partial_sums[:, depth-1, i + j*num_bits:i + (j+1)*num_bits]).unsqueeze(2).detach().clone()
                    enc_chunks.append(chunk)
                if info_bits_present:
                    concatenated_chunks = torch.cat(enc_chunks, 2)
                    if 'KO' in self.args.encoder_type:
                        if self.shared:
                            output = torch.cat([self.gnet_dict[depth](concatenated_chunks), partial_sums[:, depth-1, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                        else:
                            # bit position of the previous depth.
                            output = torch.cat([self.gnet_dict[depth][bit_position](concatenated_chunks), partial_sums[:, depth-1, i + (ell-1)*num_bits:i + (ell)*num_bits].unsqueeze(2)], dim=2)
                        output = output.permute(0,2,1).reshape(llrs.shape[0], -1, 1).squeeze(2)
                    else:
                        output = self.encode_chunks_plotkin(enc_chunks, ell)
                else:
                    output = self.encode_chunks_plotkin(enc_chunks, ell)
                partial_sums[:, depth, i : i + num_bits*ell] = output.clone()

                return decoded_llrs, partial_sums
            else: # encoding not required for last level - we have already decoded all bits.
                return decoded_llrs, partial_sums


    def kernel_decode(self, ell, fnet_dict, noisy_code, info_positions = None):
        input_shape = noisy_code.shape[-1]
        noisy_code = noisy_code.unsqueeze(2)
        assert input_shape == ell
        u = torch.ones(noisy_code.shape[0], self.N, dtype=torch.float).to(self.device)
        decoded_llrs = self.infty*torch.ones(noisy_code.shape[0], self.N, device = noisy_code.device)
        half_index = 1
        dec_chunks = [noisy_code[:, (j)*half_index:(j+1)*half_index].clone() for j in range(ell)]

        for current_position in range(ell):
            if current_position > 0:
                if self.args.onehot:
                    prev_decoded = get_onehot(u[:, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).clone().sign()).detach().clone()
                else:
                    prev_decoded = u[:, (current_position -1)*half_index:(current_position)*half_index].unsqueeze(2).clone()
                dec_chunks.append(prev_decoded)
            if current_position in info_positions:
                if current_position in info_positions:
                    concatenated_chunks = torch.cat(dec_chunks, 2)
                    Lu = fnet_dict[current_position](concatenated_chunks)
                    decoded_llrs[:, current_position] = Lu.squeeze(2).squeeze(1)
                    u_hat = torch.tanh(Lu/2).squeeze(2)
                    u[:, current_position] = u_hat.squeeze(1)
        return decoded_llrs[:, info_positions], u[:, info_positions]

    def kernel_parallel_decode(self, ell, fnet_dict, noisy_code, info_positions = None):
        input_shape = noisy_code.shape[-1]
        noisy_code = noisy_code.unsqueeze(2)
        assert input_shape == ell
        u = torch.ones(noisy_code.shape[0], self.N, dtype=torch.float).to(self.device)
        decoded_llrs = self.infty*torch.ones(noisy_code.shape[0], self.N, device = noisy_code.device)
        half_index = 1
        dec_chunks = torch.cat([noisy_code[:, (j)*half_index:(j+1)*half_index].clone() for j in range(ell)], 2)

        decoded_llrs = fnet_dict(dec_chunks).squeeze(1)
        u = torch.tanh(decoded_llrs/2).squeeze(1)
        return decoded_llrs[:, info_positions], u[:, info_positions]
