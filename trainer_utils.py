import torch
from utils import moving_average
import matplotlib.pyplot as plt
import os

def save_model(polar, iter, results_save_path, best = False):
    torch.save([polar.fnet_dict, polar.gnet_dict, polar.depth_map], os.path.join(results_save_path, 'Models/fnet_gnet_{}.pt'.format(iter)))
    if iter > 1:
        torch.save([polar.fnet_dict, polar.gnet_dict, polar.depth_map], os.path.join(results_save_path, 'Models/fnet_gnet_{}.pt'.format('final')))
    if best:
        torch.save([polar.fnet_dict, polar.gnet_dict, polar.depth_map], os.path.join(results_save_path, 'Models/fnet_gnet_{}.pt'.format('best')))

def plot_stuff(bers_enc, losses_enc, bers_dec, losses_dec, results_save_path):
    plt.figure()
    plt.plot(bers_enc, label = 'BER')
    plt.plot(moving_average(bers_enc, n=10), label = 'BER moving avg')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.title('Training BER ENC')
    plt.savefig(os.path.join(results_save_path,'training_ber_enc.png'))
    plt.close()

    plt.figure()
    plt.plot(losses_enc, label = 'Losses')
    plt.plot(moving_average(losses_enc, n=10), label='Losses moving avg')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.title('Training loss ENC')
    plt.savefig(os.path.join(results_save_path ,'training_losses_enc.png'))
    plt.close()

    plt.figure()
    plt.plot(bers_dec, label = 'BER')
    plt.plot(moving_average(bers_dec, n=10), label = 'BER moving avg')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.title('Training BER DEC')
    plt.savefig(os.path.join(results_save_path,'training_ber_dec.png'))
    plt.close()

    plt.figure()
    plt.plot(losses_dec, label = 'Losses')
    plt.plot(moving_average(losses_dec, n=10), label='Losses moving avg')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.title('Training loss DEC')
    plt.savefig(os.path.join(results_save_path ,'training_losses_dec.png'))
    plt.close()