import torch
import torch.nn.functional as F
import numpy as np
from utils import errors_ber, errors_bler, dec2bitarray, snr_db2sigma
import time


def train(args, polar, optimizer, scheduler, batch_size, train_snr, train_iters, criterion, device, info_positions, binary = False, noise_type = 'awgn'):

    if args.N == polar.ell:
        assert len(info_positions) == args.K
        kernel = True 
    else:
        kernel = False

    for iter in range(train_iters):
        if batch_size > args.small_batch_size:
            small_batch_size = args.small_batch_size 
        else:
            small_batch_size = batch_size

        num_batches = batch_size // small_batch_size
        for ii in range(num_batches):
            msg_bits = 1 - 2*(torch.rand(small_batch_size, args.K) > 0.5).float().to(device)
            if args.encoder_type == 'polar':
                codes = polar.encode_plotkin(msg_bits)
            elif 'KO' in args.encoder_type:
                if kernel:
                    codes = polar.kernel_encode(args.kernel_size, polar.gnet_dict[1][0], msg_bits, info_positions, binary = binary)
                else:
                    codes = polar.deeppolar_encode(msg_bits, binary = binary)

            noisy_codes = polar.channel(codes, train_snr, noise_type)

            if 'KO' in args.decoder_type:
                if kernel:
                    if args.decoder_type == 'KO_parallel':
                        decoded_llrs, decoded_bits = polar.kernel_parallel_decode(args.kernel_size, polar.fnet_dict[1][0], noisy_codes, info_positions)
                    else:
                        decoded_llrs, decoded_bits = polar.kernel_decode(args.kernel_size, polar.fnet_dict[1][0], noisy_codes, info_positions)
                else:
                    decoded_llrs, decoded_bits = polar.deeppolar_decode(noisy_codes)
            elif args.decoder_type == 'SC':
                decoded_llrs, decoded_bits = polar.sc_decode_new(noisy_codes, train_snr)

            if 'BCE' in args.loss or args.loss == 'focal':
                loss = criterion(decoded_llrs, 0.5 * msg_bits.to(polar.device) + 0.5)
            else:
                loss = criterion(torch.tanh(0.5*decoded_llrs), msg_bits.to(polar.device))
            
            if args.regularizer == 'std':
                if args.K == 1:
                    loss += args.regularizer_weight * torch.std(codes, dim=1).mean()
                elif args.K == 2:
                    loss += args.regularizer_weight * (0.5*torch.std(codes[:, ::2], dim=1).mean() + .5*torch.std(codes[:, 1::2], dim=1).mean())
            elif args.regularizer == 'max_deviation':
                if args.K == 1:
                    loss += args.regularizer_weight * torch.amax(torch.abs(codes - codes.mean(dim=1, keepdim=True)), dim=1).mean()
                elif args.K == 2:
                    loss += args.regularizer_weight * (0.5*torch.amax(torch.abs(codes[:, ::2] - codes[:, ::2].mean(dim=1, keepdim=True)), dim=1).mean() + .5*torch.amax(torch.abs(codes[:, 1::2] - codes[:, 1::2].mean(dim=1, keepdim=True)), dim=1).mean())
            elif args.regularizer == 'polar':
                loss += args.regularizer_weight * F.mse_loss(codes, polar.encode_plotkin(msg_bits))

            loss = loss/num_batches
            loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    train_ber = errors_ber(decoded_bits.sign(), msg_bits.to(polar.device)).item()

    return loss.item(), train_ber

def deeppolar_full_test(args, polar, KO, snr_range, device, info_positions, binary=False, num_errors=100, noise_type = 'awgn'):
    bers_KO_test = [0. for _ in snr_range]
    blers_KO_test = [0. for _ in snr_range]

    bers_SC_test = [0. for _ in snr_range]
    blers_SC_test = [0. for _ in snr_range]

    kernel = args.N == KO.ell

    print(f"TESTING until {num_errors} block errors")
    for snr_ind, snr in enumerate(snr_range):
        total_block_errors_SC = 0
        total_block_errors_KO = 0
        batches_processed = 0

        sigma = snr_db2sigma(snr)  # Assuming SNR is given in dB and noise variance is derived from it

        try:
            while min(total_block_errors_SC, total_block_errors_KO) <= num_errors:
                msg_bits = 2 * (torch.rand(args.test_batch_size, args.K) < 0.5).float() - 1
                msg_bits = msg_bits.to(device)
                polar_code = polar.encode_plotkin(msg_bits)

                if 'KO' in args.encoder_type:
                    if kernel:
                        KO_polar_code = KO.kernel_encode(args.kernel_size, KO.gnet_dict[1][0], msg_bits, info_positions, binary=binary)
                    else:
                        KO_polar_code = KO.deeppolar_encode(msg_bits, binary=binary)

                noisy_code = polar.channel(polar_code, snr, noise_type)
                noise = noisy_code - polar_code
                noisy_KO_code = KO_polar_code + noise if 'KO' in args.encoder_type else noisy_code

                SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
                ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign()).item()
                bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign()).item()
                total_block_errors_SC += int(bler_SC*args.test_batch_size)
                if 'KO' in args.decoder_type:
                    if kernel:
                        if args.decoder_type == 'KO_parallel':
                            KO_llrs, decoded_KO_msg_bits = KO.kernel_parallel_decode(args.kernel_size, KO.fnet_dict[1][0], noisy_KO_code, info_positions)
                        else:
                            KO_llrs, decoded_KO_msg_bits = KO.kernel_decode(args.kernel_size, KO.fnet_dict[1][0], noisy_KO_code, info_positions)
                    else:
                        KO_llrs, decoded_KO_msg_bits = KO.deeppolar_decode(noisy_KO_code)
                else:  # if SC is also used for KO
                    KO_llrs, decoded_KO_msg_bits = KO.sc_decode_new(noisy_KO_code, snr)

                ber_KO = errors_ber(msg_bits, decoded_KO_msg_bits.sign()).item()
                bler_KO = errors_bler(msg_bits, decoded_KO_msg_bits.sign()).item()
                total_block_errors_KO += int(bler_KO*args.test_batch_size)

                batches_processed += 1

                # Update accumulative results for logging
                bers_KO_test[snr_ind] += ber_KO
                bers_SC_test[snr_ind] += ber_SC
                blers_KO_test[snr_ind] += bler_KO
                blers_SC_test[snr_ind] += bler_SC

                # Real-time logging for progress, updating in-place
                print(f"SNR: {snr} dB, Sigma: {sigma:.5f}, SC_BER: {bers_SC_test[snr_ind]/batches_processed:.6f}, SC_BLER: {blers_SC_test[snr_ind]/batches_processed:.6f}, KO_BER: {bers_KO_test[snr_ind]/batches_processed:.6f}, KO_BLER: {blers_KO_test[snr_ind]/batches_processed:.6f}, Batches: {batches_processed}", end='\r')

        except KeyboardInterrupt:
            # print("\nInterrupted by user. Finalizing current SNR...")
            pass

        # Normalize cumulative metrics by the number of processed batches for accuracy
        bers_KO_test[snr_ind] /= (batches_processed + 0.00000001)
        bers_SC_test[snr_ind] /= (batches_processed + 0.00000001)
        blers_KO_test[snr_ind] /= (batches_processed + 0.00000001)
        blers_SC_test[snr_ind] /= (batches_processed + 0.00000001)
        print(f"SNR: {snr} dB, Sigma: {sigma:.5f}, SC_BER: {bers_SC_test[snr_ind]:.6f}, SC_BLER: {blers_SC_test[snr_ind]:.6f}, KO_BER: {bers_KO_test[snr_ind]:.6f}, KO_BLER: {blers_KO_test[snr_ind]:.6f}")

    return bers_SC_test, blers_SC_test, bers_KO_test, blers_KO_test


       
