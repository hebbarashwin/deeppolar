# DeepPolar codes
Code for "DeepPolar codes", ICML 2024 


## Requirements

- Python 3.6 or later
- Relevant Python libraries as specified in `requirements.txt`

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Best results are obtained by pretraining kernels using curriculum training and initializing the network using these pretrained kernels. (training from scratch may work too)
An exemplar kernel has been provided. Command to run:

```bash
python -u main.py --N 256 --K 37 --model_save_per 100 --enc_train_iters 20 --dec_train_iters 200 --full_iters 2000 --enc_train_snr 0 --dec_train_snr -2 --enc_hidden_size 64 --dec_hidden_size 128 --enc_lr 0.0001 --dec_lr 0.0001  --weight_decay 0 --test_snr_start -5 --test_snr_end -1 --snr_points 5 -ell 16 --batch_size 20000 --id run1 --kernel_load_path Polar_Results/curriculum/final_kernels/16_normal_polar_eh64_dh128_selu
```

- `N`, `K`: Code parameters
- `model_save_per`: Frequency of saving the trained models.
- `enc_train_iters`, `dec_train_iters`: Number of training iterations for the encoder and decoder.
- `full_iters`: Total iterations for full training cycles.
- `enc_train_snr`, `dec_train_snr`: Signal to noise ratios for encoder and decoder training.
- `enc_hidden_size`, `dec_hidden_size`: Hidden layer sizes for the encoder and decoder.
- `enc_lr`, `dec_lr`: Learning rates for the encoder and decoder.
- `test_size`: Number of samples used in testing.
- `test_snr_start`, `test_snr_end`: Start and end signal to noise ratios for testing.
- `snr_points`: Number of SNR points.
- `-ell`, Kernel size; \sqrt{N} works best
- `batch_size`: Batch size
- `id`: Identifier for the run.
- `kernel_load_path`: Path to load specific model kernels.


The kernels can be pretrained, for example by running 
```bash
bash pretrain.sh
```

(More details will be added soon.)