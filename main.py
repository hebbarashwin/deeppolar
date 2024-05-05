import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import os
import time
import matplotlib 
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from deeppolar import DeepPolar
from polar import PolarCode, get_frozen
from trainer import train, deeppolar_full_test
from trainer_utils import save_model, plot_stuff
from collections import defaultdict
from itertools import combinations
from utils import snr_db2sigma, pairwise_distances

import random
import numpy as np
from tqdm import tqdm
import sys
import csv


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(input_string):
    """Converts a comma-separated string into a list of integers."""
    try:
        if len(input_string) == 0 :
            return []
        return [int(item) for item in input_string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"List must contain integers, got '{input_string}'")

# NN definition - define_kernel_nns for kernel, define_and_load_nns for general NN
# Encoding kernel_encode or deeppolar_encode
# Decoding kernel_decode, or deeppolar_decode

def get_args():
    parser = argparse.ArgumentParser(description='DeepPolar codes')

    # General parameters
    parser.add_argument('--id', type=str, default=None, help='ID: optional, to run multiple runs of same hyperparameters') #Will make a folder like init_932 , etc.
    parser.add_argument('--test', dest = 'test', default=False, action='store_true', help='Testing?')
    parser.add_argument('--pairwise', dest = 'pairwise', default=False, action='store_true', help='Plot codeword pairwise distances')
    parser.add_argument('--epos', dest = 'epos', default=False, action='store_true', help='Plot error positions')
    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true', help='Helper to load functions on jupyter')
    parser.add_argument('--gpu', type=int, default=-2, help='gpus used for training - e.g 0,1,3. -2 for cuda, -1 for cpu')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--anomaly', dest = 'anomaly', default=False, action='store_true', help='enable anomaly detection')
    parser.add_argument("--dataparallel", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Use Dataparallel")

    # Code parameters
    parser.add_argument('--N', type=int, default=256, help = 'Block length')#, choices=[4, 8, 16, 32, 64, 128], help='Polar code parameter N')
    parser.add_argument('--K', type=int, default=37, help = 'Message size')#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')
    parser.add_argument('--rate_profile', type=str, default='polar', choices=['RM', 'polar', 'sorted', 'last', 'rev_polar', 'custom'], help='Polar rate profiling')
    # parser.add_argument('--target_K', type=int, default=None)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')
    parser.add_argument('-ell', '--kernel_size', type=int, default=16, help = 'Kernel size')
    parser.add_argument('--polar_depths', type=parse_int_list, default = '',help='A comma-separated list of integers')
    parser.add_argument('--last_ell', type=int, default=None, help='use kernel last_ell last layer')
    parser.add_argument('--infty', type=float, default=1000., help = 'Infinity value (used for frozen position LLR in polar dec)')
    parser.add_argument('--lse', type=str, default='minsum', choices=['minsum', 'lse'], help='LSE function for polar SC decoder')
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true', help='polar code sc decoding hard decision?')

    # DeepPolar parameters
    parser.add_argument('--encoder_type', type=str, default='KO', choices=['KO', 'scaled', 'polar'], help='Type of encoding')
    parser.add_argument('--decoder_type', type=str, default='KO', choices=['KO', 'SC', 'KO_parallel', 'KO_last_parallel'], help='Type of encoding')
    parser.add_argument('--enc_activation', type=str, default='selu', choices=['selu', 'leaky_relu', 'gelu', 'silu', 'elu', 'mish', 'identity'], help='Activation function')
    parser.add_argument('--dec_activation', type=str, default='selu', choices=['selu', 'leaky_relu', 'gelu', 'silu', 'elu', 'mish', 'identity'], help='Activation function')
    parser.add_argument('--dropout_p', type=float, default=0.)
    parser.add_argument('--dec_hidden_size', type=int, default=128, help='neural network size')
    parser.add_argument('--enc_hidden_size', type=int, default=64, help='neural network size')
    parser.add_argument('-fd', '--f_depth', type=int, default=3, help='decoder neural network depth')
    parser.add_argument('-gd', '--g_depth', type=int, default=3, help='encoder neural network depth')
    parser.add_argument('-gsd', '--g_skip_depth', type=int, default=1, help='encoder neural network depth')
    parser.add_argument('-gsl', '--g_skip_layer', type=int, default=1, help='encoder neural network depth')
    parser.add_argument("--onehot", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Use onehot representation of prev_decoded_bits?")
    parser.add_argument("--shared", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Share weights across depth?")
    parser.add_argument("--skip", type=str2bool, nargs='?',
                            const=True, default=True,
                            help="Use skip")
    parser.add_argument("--use_norm", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Use norm")
    parser.add_argument("--binary", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="")

    # Training parameters
    parser.add_argument('-fi', '--full_iters', type=int, default=20000, help='full iterations')
    parser.add_argument('-ei', '--enc_train_iters', type=int, default=20, help='encoder iterations') #50
    parser.add_argument('-di', '--dec_train_iters', type=int, default=200, help='decoder iterations') #500

    parser.add_argument('--enc_train_snr', type=float, default=0., help='snr at enc are trained')
    parser.add_argument('--dec_train_snr', type=float, default=-2., help='snr at dec are trained')


    parser.add_argument('--initialization', type=str, default='random', choices=['random', 'zeros'], help='initialization')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'RMS', 'SGD', 'AdamW'], help='optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--loss', type=str, default='BCE', choices=['MSE', 'BCE', 'BCE_reg', 'L1', 'huber', 'focal', 'BCE_bler'], help='loss function')
    parser.add_argument('--dec_lr', type=float, default=0.0003, help='Decoder Learning rate')
    parser.add_argument('--enc_lr', type=float, default=0.0003, help='Encoder Learning rate')

    parser.add_argument('--regularizer', type=str, default=None, choices=['std', 'max_deviation','polar'], help='regularize the kernel pretraining')
    parser.add_argument('-rw', '--regularizer_weight', type=float, default=0.001)

    parser.add_argument('--scheduler', type=str, default=None, choices = ['reduce', '1cycle'],help='size of the batches')
    parser.add_argument('--scheduler_patience', type=int, default=None, help='size of the batches')

    parser.add_argument('--small_batch_size', type=int, default=20000, help='size of the batches')
    parser.add_argument('--batch_size', type=int, default=20000, help='size of the batches')
    parser.add_argument("--batch_schedule", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Batch scheduler")
    parser.add_argument('--batch_patience', type=int, default=50, help='patience')
    parser.add_argument('--batch_factor', type=int, default=2, help='patience')
    parser.add_argument('--min_batch_size', type=int, default=500, help='patience')
    parser.add_argument('--max_batch_size', type=int, default=50000, help='patience')


    parser.add_argument('--noise_type', type=str, default='awgn', choices=['fading', 'awgn', 'radar'], help='loss function')
    parser.add_argument('--radar_power', type=float, default=None, help='snr at dec are trained')
    parser.add_argument('--radar_prob', type=float, default=0.1, help='snr at dec are trained')


    # TESTING parameters
    parser.add_argument('--model_save_per', type=int, default=100, help='num of episodes after which model is saved')
    parser.add_argument('--test_snr_start', type=float, default=-5., help='testing snr start')
    parser.add_argument('--test_snr_end', type=float, default=-1., help='testing snr end')
    parser.add_argument('--snr_points', type=int, default=5, help='testing snr num points')
    parser.add_argument('--test_batch_size', type=int, default=10000, help='size of the batches')
    parser.add_argument('--num_errors', type=int, default=100, help='Test until _ block errors')
    parser.add_argument('--model_iters', type=int, default=None, help='by default load final model, option to load a model of x episodes')
    parser.add_argument('--test_load_path', type=str, default=None, help='load test model given path')
    parser.add_argument('--save_path', type=str, default=None, help='save name')
    parser.add_argument('--load_path', type=str, default=None, help='load name')
    parser.add_argument('--kernel_load_path', type=str, default=None, help='load name')
    parser.add_argument("--no_fig", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Plot fig?")

    args = parser.parse_args()

    if args.small_batch_size > args.batch_size:
        args.small_batch_size = args.batch_size

    return args

if __name__ == '__main__':

    args = get_args()
    if not args.test:
        print(args)

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available() and args.gpu != -1:
        if args.gpu == -2:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:{0}".format(args.gpu))
    else:
        if args.gpu != 1:
            print(f"GPU device {args.gpu if args.gpu != -2 else ''} not found.")
        device = torch.device("cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ID = str(np.random.randint(100000, 999999)) if args.id is None else args.id

    if args.save_path is not None:
        results_save_path = args.save_path
    else:
        if args.encoder_type == 'polar':
            results_save_path = './Polar_Results/Polar({0},{1})/Scheme_{2}/KO_Decoder/{3}'.format(args.K, args.N, args.rate_profile, ID)
        elif 'KO' in args.encoder_type:
            if args.decoder_type == 'KO_last_parallel':
                dec = '_lp'
            else:
                dec = ''
            results_save_path = f"./Polar_Results/Polar_{args.kernel_size}({args.N},{args.K})/Scheme_{args.rate_profile}/{args.encoder_type}_Encoder{dec}_Decoder/{ID}"
        elif args.encoder_type == 'scaled':
            if args.decoder_type == 'SC':
                results_save_path = './Polar_Results/Polar({0},{1})/Scheme_{2}/Scaled_Decoder/{3}'.format(args.K, args.N, args.rate_profile, ID)
            else:
                results_save_path = './Polar_Results/Polar({0},{1})/Scheme_{2}/KO_Scaled_Decoder/{3}'.format(args.K, args.N, args.rate_profile, ID)


    ############
    ## Polar Code parameters
    ############
    K = args.K
    N = args.N

    ###############
    ### Polar code
    ##############

    ### Encoder

    if args.last_ell is not None:
        depth_map = defaultdict(int)
        n = int(np.log2(args.N // args.last_ell) // np.log2(args.kernel_size))
        for d in range(1, n+1):
            depth_map[d] = args.kernel_size
        depth_map[n+1] = args.last_ell
        assert np.prod(list(depth_map.values())) == args.N
        polar = DeepPolar(args, device, args.N, args.K, infty = args.infty, depth_map = depth_map)
    else:
        polar = DeepPolar(args, device, args.N, args.K, args.kernel_size, args.infty)

    info_inds = polar.info_positions
    frozen_inds = polar.frozen_positions

    print("Frozen positions : {}".format(frozen_inds))
    if args.only_args:
        print("Loaded args. Exiting")
        sys.exit()
    ##############
    ### Neural networks
    ##############
    ell = args.kernel_size
    if args.N == ell: # Kernel pre-training
        polar.define_kernel_nns(ell = args.kernel_size, unfrozen = polar.info_positions, fnet = args.decoder_type, gnet = args.encoder_type, shared = args.shared)
    elif args.N > ell: # Initialize full network with pretrained kernels
        polar.define_and_load_nns(ell = args.kernel_size, kernel_load_path=args.kernel_load_path, fnet = args.decoder_type, gnet = args.encoder_type, shared = args.shared, dataparallel=args.dataparallel)

    if args.binary:
        args.load_path = os.path.join(results_save_path, 'Models/fnet_gnet_final.pt')
        assert os.path.exists(args.load_path), "Model does not exist!!"
        results_save_path = os.path.join(results_save_path, 'Binary')
        os.makedirs(results_save_path, exist_ok=True)
        os.makedirs(results_save_path +'/Models', exist_ok=True)

    if args.load_path is not None:
        if args.test:
            if args.test_load_path is None:
                print("WARNING : have you used load_path instead of test_load_path?")
        else:
            checkpoint1 = torch.load(args.load_path , map_location=lambda storage, loc: storage)
            fnet_dict = checkpoint1[0]
            gnet_dict = checkpoint1[1]

            polar.load_partial_nns(fnet_dict, gnet_dict)
            print("Loaded nets from {}".format(args.load_path))

    if 'KO' in args.decoder_type:
        dec_params = []
        for i in polar.fnet_dict.keys():
            for j in polar.fnet_dict[i].keys():
                if isinstance(polar.fnet_dict[i][j], dict):
                    for k in polar.fnet_dict[i][j].keys():
                        dec_params += list(polar.fnet_dict[i][j][k].parameters())
                else:
                    dec_params += list(polar.fnet_dict[i][j].parameters())
    elif args.decoder_type == 'RNN':
        dec_params = polar.fnet_dict.parameters()
    else:
        args.dec_train_iters = 0

    if 'KO' in args.encoder_type:
        enc_params = []
        if args.shared:
            for i in polar.gnet_dict.keys():
                enc_params += list(polar.gnet_dict[i].parameters())
        else:
            for i in polar.gnet_dict.keys():
                for j in polar.gnet_dict[i].keys():
                    enc_params += list(polar.gnet_dict[i][j].parameters())
    elif args.encoder_type == 'scaled':
        enc_params = [polar.a]
        enc_optimizer = optim.Adam(enc_params, lr = args.enc_lr)
    else:
        args.enc_train_iters = 0

    if args.dec_train_iters > 0:
        if args.optim == 'Adam':
            dec_optimizer = optim.Adam(dec_params, lr = args.dec_lr, weight_decay = args.weight_decay)#, momentum=0.9, nesterov=True) #, amsgrad=True)
        elif args.optim == 'SGD':
            dec_optimizer = optim.SGD(dec_params, lr = args.dec_lr, weight_decay = args.weight_decay)#, momentum=0.9, nesterov=True) #, amsgrad=True)
        elif args.optim == 'RMS':
            dec_optimizer = optim.RMSprop(dec_params, lr = args.dec_lr, weight_decay = args.weight_decay)#, momentum=0.9, nesterov=True) #, amsgrad=True)
        if args.scheduler == 'reduce':
            dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dec_optimizer, 'min', patience = args.scheduler_patience)  
        elif args.scheduler == '1cycle':
            dec_scheduler = optim.lr_scheduler.OneCycleLR(dec_optimizer, max_lr = args.dec_lr, total_steps=args.dec_train_iters*args.full_iters)  
        else:
            dec_scheduler = None

    if args.enc_train_iters > 0:
        enc_optimizer = optim.Adam(enc_params, lr = args.enc_lr)#, momentum=0.9, nesterov=True) #, amsgrad=True)
        if args.scheduler == 'reduce':
            enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(enc_optimizer, 'min', patience = args.scheduler_patience)  
        elif args.scheduler == '1cycle':
            enc_scheduler = optim.lr_scheduler.OneCycleLR(enc_optimizer, max_lr = args.enc_lr, total_steps=args.enc_train_iters*args.full_iters) 
        else:
            enc_scheduler = None
    
        


    if 'BCE' in args.loss:
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'L1':
        criterion = nn.L1Loss()
    elif args.loss == 'huber':
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss() 

    info_positions = polar.info_positions
    if not args.test:
        bers_enc = []
        losses_enc = []
        bers_dec = []
        losses_dec = []
        train_ber_dec = 0.
        train_ber_enc = 0.
        loss_dec = 0.
        loss_enc = 0.
        # val_bers = []
        os.makedirs(results_save_path, exist_ok=True)
        os.makedirs(results_save_path +'/Models', exist_ok=True)

        # Create CSV at the beginning of training
        save_path_id = random.randint(100000, 999999)
        with open(os.path.join(results_save_path, f'training_results_{save_path_id}.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Step', 'Loss', 'BER'])

            # save args in a json file
            

        ##############
        ### Optimizers
        ##############

        print("Need to save for:", args.model_save_per)
        if not args.batch_schedule:
            batch_size = args.batch_size 
        else:
            batch_size = args.min_batch_size 
            best_batch_ber = 10.
            best_batch_iter = 0
        try:
            best_ber = 10.
            for iter in range(1, args.full_iters + 1):
                start_time = time.time()

                if not args.batch_schedule:
                    batch_size = args.batch_size 
                elif batch_size != args.max_batch_size:
                    if iter - best_batch_iter > args.batch_patience:
                        batch_size = min(batch_size * 2, args.max_batch_size)
                        print(f"Increased batch size to {batch_size}")
                        best_batch_ber = train_ber_enc
                        best_batch_iter = iter                        
                if 'KO' in args.decoder_type or args.decoder_type == 'RNN':
                    # Train decoder
                    loss_dec, train_ber_dec = train(args, polar, dec_optimizer, dec_scheduler if args.scheduler == '1cycle' else None, batch_size, args.dec_train_snr, args.dec_train_iters, criterion, device, info_positions, binary = args.binary, noise_type = args.noise_type)
                    if args.scheduler_patience is not None:
                        dec_scheduler.step(loss_dec)                    
                    bers_dec.append(train_ber_dec)
                    losses_dec.append(loss_dec)
                if 'KO' in args.encoder_type:
                    # Train encoder
                    loss_enc, train_ber_enc = train(args, polar, enc_optimizer, enc_scheduler if args.scheduler == '1cycle' else None, batch_size, args.enc_train_snr, args.enc_train_iters, criterion, device, info_positions, binary = args.binary, noise_type = args.noise_type)
                    if args.scheduler_patience is not None:
                        enc_scheduler.step(loss_enc)                    
                    bers_enc.append(train_ber_enc)
                    losses_enc.append(loss_enc)  
                

                if args.batch_schedule and train_ber_enc < best_batch_ber:
                    best_batch_ber = train_ber_enc
                    best_batch_iter = iter
                    print(f'Best BER {best_batch_ber} at {best_batch_iter}')

                # Save to CSV
                with open(os.path.join(results_save_path, f'training_results_{save_path_id}.csv'), 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([iter, loss_enc, train_ber_enc, loss_dec, train_ber_dec])
                if iter % 10 == 1:
                    print(f"[{iter}/{args.full_iters}] At {args.dec_train_snr} dB, Train Loss: {loss_dec} Train BER {train_ber_dec}, \
                          \n [{iter}/{args.full_iters}] At {args.enc_train_snr} dB, Train Loss: {loss_enc} Train BER {train_ber_enc}")
                    print("Time for one full iteration is {0:.4f} minutes. save ID = {1}".format((time.time() - start_time)/60, ID))
                if iter % args.model_save_per == 0 or iter == 1:
                    if train_ber_enc < best_ber:
                        best_ber = train_ber_enc
                        best = True 
                    else:
                        best = False
                    save_model(polar, iter, results_save_path, best = best)
                    plot_stuff(bers_enc, losses_enc, bers_dec, losses_dec, results_save_path)
            save_model(polar, iter, results_save_path)
            plot_stuff(bers_enc, losses_enc, bers_dec, losses_dec, results_save_path)

        except KeyboardInterrupt:

            save_model(polar, iter, results_save_path)
            plot_stuff(bers_enc, losses_enc, bers_dec, losses_dec, results_save_path)

            print("Exited and saved")


    print("TESTING")
    times = []
    results_load_path = results_save_path


    if args.model_iters is not None:
        checkpoint1 = torch.load(results_save_path +'/Models/fnet_gnet_{}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)
    elif args.test_load_path is not None:
        checkpoint1 = torch.load(args.test_load_path , map_location=lambda storage, loc: storage)
    else:
        checkpoint1 = torch.load(results_load_path +'/Models/fnet_gnet_final.pt', map_location=lambda storage, loc: storage)

    fnet_dict = checkpoint1[0]
    gnet_dict = checkpoint1[1]

    polar.load_nns(fnet_dict, gnet_dict, shared = args.shared)

    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start)* 1.0 /  (args.snr_points-1)
        snr_range = [snrs_interval* item + args.test_snr_start for item in range(args.snr_points)]

    start_time = time.time()

    # For polar code testing.
    args2 = argparse.Namespace(**vars(args))
    args2.ell = 2
    Frozen = get_frozen(N, K, args2.rate_profile)
    Frozen.sort()
    polar_l_2 = PolarCode(int(np.log2(N)), args.K, Fr=Frozen, infty = args.infty, hard_decision=args.hard_decision)


    if args.pairwise:
        codebook_size = 1000
        all_msg_bits = 2 * (torch.rand(codebook_size, args.K, device = device) < 0.5).float() - 1
        deeppolar_codebook = polar.deeppolar_encode(all_msg_bits)
        polar_codebook = polar_l_2.encode_plotkin(all_msg_bits)
        gaussian_codebook = F.normalize(torch.randn(codebook_size, args.N), p=2, dim=1)*np.sqrt(args.N)

        from scipy import stats
        w_statistic_deeppolar, p_value_deeppolar = stats.shapiro(deeppolar_codebook.detach().cpu().numpy())
        w_statistic_gaussian, p_value_gaussian = stats.shapiro(gaussian_codebook.detach().cpu().numpy())
        w_statistic_polar, p_value_polar = stats.shapiro(polar_codebook.detach().cpu().numpy())

        print(f"Deeppolar Shapiro test W = {w_statistic_deeppolar}, p-value = {p_value_deeppolar}")
        print(f"Gaussian Shapiro test W = {w_statistic_gaussian}, p-value = {p_value_gaussian}")
        print(f"Polar Shapiro test W = {w_statistic_polar}, p-value = {p_value_polar}")

        dists_deeppolar, md_deeppolar = pairwise_distances(deeppolar_codebook)
        dists_polar, md_polar = pairwise_distances(polar_codebook)
        dists_gaussian, md_gaussian = pairwise_distances(gaussian_codebook)

        # Function to calculate and plot PDF
        def plot_pdf(data, label, bins=30, alpha=0.5):
            counts, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, counts, label=label, alpha=alpha)

        # Plotting PDF for each list
        plt.figure()
        plot_pdf(dists_deeppolar, 'Neural', 300)
        # plot_pdf(dists_polar, 'Polar', 300)
        plot_pdf(dists_gaussian, 'Gaussian', 300)

        # Adding labels and title
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.title(f'Pairwise Distances - N = {args.N}, K = {args.K}')
        plt.legend()

        # Show the plot
        plt.savefig(os.path.join(results_save_path, f"hists_N{args.N}_K{args.K}_{args.id}_2.pdf"))

    if args.epos:
        from collections import OrderedDict, Counter

        def get_epos(k1, k2):
            # return counter for bit ocations of first-errors
            bb = torch.ne(k1.cpu().sign(), k2.cpu().sign())
            # inds = torch.nonzero(bb)[:, 1].numpy()
            idx = []
            for ii in range(bb.shape[0]):
                try:
                    iii = list(bb.cpu().float().numpy()[ii]).index(1)
                    idx.append(iii)
                except:
                    pass
            counter = Counter(idx)
            ordered_counter = OrderedDict(sorted(counter.items()))
            return ordered_counter

        with torch.no_grad():
            for (k, msg_bits) in enumerate(Test_Data_Generator):
                msg_bits = msg_bits.to(device)
                polar_code = polar_l_2.encode_plotkin(msg_bits)
                noisy_code = polar.channel(polar_code, args.dec_train_snr)
                noise = noisy_code - polar_code
                deeppolar_code = polar.deeppolar_encode(msg_bits)
                noisy_deeppolar_code = deeppolar_code + noise
                SC_llrs, decoded_SC_msg_bits = polar_l_2.sc_decode_new(noisy_code, args.dec_train_snr)
                deeppolar_llrs, decoded_deeppolar_msg_bits = polar.deeppolar_decode(noisy_deeppolar_code)

                if k == 0:
                    epos_deeppolar = get_epos(msg_bits, decoded_deeppolar_msg_bits.sign())
                    epos_SC = get_epos(msg_bits, decoded_SC_msg_bits.sign())
                else:
                    epos_deeppolar1 = get_epos(msg_bits, decoded_deeppolar_msg_bits.sign())
                    epos_SC1 = get_epos(msg_bits, decoded_SC_msg_bits.sign())
                    epos_deeppolar = epos_deeppolar + epos_deeppolar1
                    epos_SC = epos_SC + epos_SC1

            print(f"epos_deeppolar: {epos_deeppolar}")
            print(f"EPOS_SC: {epos_SC}")



    start = time.time()
    bers_SC_test, blers_SC_test, bers_deeppolar_test, blers_deeppolar_test = deeppolar_full_test(args, polar_l_2, polar, snr_range, device, info_positions, binary = args.binary, noise_type = args.noise_type, num_errors = args.num_errors)
    print("Test SNRs : {}\n".format(snr_range))
    print(f"Test Sigmas : {[snr_db2sigma(s) for s in snr_range]}\n")
    print("BERs of DeepPolar: {0}".format(bers_deeppolar_test))
    print("BERs of SC decoding: {0}".format(bers_SC_test))
    print("BLERs of DeepPolar: {0}".format(blers_deeppolar_test))
    print("BLERs of SC decoding: {0}".format(blers_SC_test))
    print(f"time = {(time.time() - start)/60} minutes")
    ## BER
    plt.figure(figsize = (12,8))

    ok = 0
    plt.semilogy(snr_range, bers_deeppolar_test, label="DeepPolar", marker='*', linewidth=1.5)

    plt.semilogy(snr_range, bers_SC_test, label="SC decoder", marker='^', linewidth=1.5)

    ## BLER
    plt.semilogy(snr_range, blers_deeppolar_test, label="DeepPolar (BLER)", marker='*', linewidth=1.5, linestyle='dashed')

    plt.semilogy(snr_range, blers_SC_test, label="SC decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')

    plt.grid()
    plt.xlabel("SNR (dB)", fontsize=16)
    plt.ylabel("Error Rate", fontsize=16)
    if args.enc_train_iters > 0:
        plt.title("PolarC({2}, {3}): DeepPolar trained at Dec_SNR = {0} dB, Enc_SNR = {1}dB".format(args.dec_train_snr, args.enc_train_snr, args.K,args.N))
    else:
        plt.title("Polar({1}, {2}): DeepPolar trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
    plt.legend(prop={'size': 15})
    if args.test_load_path is not None:
        os.makedirs('Polar_Results/figures', exist_ok=True)
        fig_save_path = 'Polar_Results/figures/new_plot_DeepPolar.pdf'
    else:
        fig_save_path = results_load_path + f"/Step_{args.model_iters if args.model_iters is not None else 'final'}{'_binary' if args.binary else ''}.pdf"
    if not args.no_fig:
        plt.savefig(fig_save_path)

    plt.close()
