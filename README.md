# DeepPolar codes
Code for "[DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep Learning](https://arxiv.org/abs/2402.08864)", ICML 2024 

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/hebbarashwin/deeppolar.git
cd deeppolar
```

Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Best results are obtained by pretraining kernels using curriculum training and initializing the network using these pretrained kernels. (training from scratch may work too)
An exemplar kernel has been provided. Command to run:
(You can set --id for different runs.)

```bash
python -u main.py --N 256 --K 37  -ell 16 --model_save_per 100 --enc_train_iters 20 --dec_train_iters 200 --full_iters 2000 --enc_train_snr 0 --dec_train_snr -2 --enc_hidden_size 64 --dec_hidden_size 128 --enc_lr 0.0001 --dec_lr 0.0001  --weight_decay 0 --test_snr_start -5 --test_snr_end -1 --snr_points 5 --batch_size 20000 --id run1 --kernel_load_path Polar_Results/curriculum/final_kernels/16_normal_polar_eh64_dh128_selu --gpu -2
```

- `N`, `K`: Code parameters
- `-ell`, Kernel size; \sqrt{N} works best
- `kernel_load_path`: Path to load specific model kernels. (if training from scratch, don't set this flag)
- `enc_train_iters`, `dec_train_iters`: Number of training iterations for the encoder and decoder.
- `full_iters`: Total iterations for full training cycles.
- `id`: Identifier for the run.
- `model_save_per`: Frequency of saving the trained models.
- `gpu` : -2 : cuda, -1 : cpu, 0/1/2/3 : specific gpu


The kernels can be pretrained, for example by running 
```bash
bash pretrain.sh
```
(Typically we don't need to train each kernel for as many iterations as this script.)

Testing

```bash
python -u main.py --N 256 --K 37 -ell 16 --enc_hidden_size 64 --dec_hidden_size 128 --test_snr_start -5 --test_snr_end -1 --snr_points 5 --test_batch_size 10000 --id run1 --weight_decay 0. --num_errors 100 --test
```

(More details will be added soon.)
- Finetuning with increasingly large batch sizes improves high-SNR performance.
- BER gain can be traded off for BLER by finetuning with a BLER surrogate loss.