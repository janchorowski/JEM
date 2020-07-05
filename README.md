# JEM (Joint Energy-Based Models) for Semi-Supervised Classification

Code for the paper "Joint Energy-Based Models for Semi-Supervised Classification". 

This work builds upon [the original JEM work](https://github.com/wgrathwohl/JEM).

Many thanks to my amazing co-authors and mentors: [Jörn-Henrick Jacobsen](https://jhjacobsen.github.io/) and [Will Grathwohl](http://www.cs.toronto.edu/~wgrathwohl/). 


## Example Commands for Running Experiments

Below commands are for running JEM. Use ```--vat``` flag to run VAT, and optionally change ```--vat_eps 3.0```. Set ```--p_x_weight 0.0``` instead to run just the baseline classifier (with ```--dropout --batch_norm``` for regularization).

### Toy Datasets
To train and evaluate JEM on the rings (concentric circles) dataset:
```markdown
python train_nn_ebm.py --labels_per_class 2 --rings_noise 0.03 --l2_energy_reg 0.0002 --l2_energy_reg_neg --n_rings_data 1000 --lr .001 --use_nn --batch_size 20 --dataset rings --n_valid 200 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --data_seed 20 --plot_uncond --warmup_iters 10 --weight_decay .0005 --sgld_lr .00125 --sgld_std .05 --temper_init 1. --ul_batch_size 100 --viz_every 10
```
Use ```--dataset moons``` instead for moons dataset:

```markdown
python train_nn_ebm.py --labels_per_class 2 --moons_noise 0.1 --l2_energy_reg 0.001 --l2_energy_reg_neg --n_moons_data 1000 --lr .001 --use_nn --batch_size 20 --dataset moons --n_valid 200 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --data_seed 20 --plot_uncond --warmup_iters 10 --save_dir . --weight_decay .0005 --sgld_lr .00125 --sgld_std .05 --temper_init 1. --ul_batch_size 100 --viz_every 10
```

### MNIST

```markdown
python train_nn_ebm.py --lr .0002 --use_nn --nn_hidden_size 500 --nn_extra_layers 2 --batch_size 50 --dataset mnist --n_valid 5000 --labels_per_class 10 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --warmup_iters 1000 --n_steps 40
```

### SVHN

```markdown
python train_nn_ebm.py --lr .0002 --use_nn --nn_hidden_size 1000 --nn_extra_layers 1 --batch_size 100 --dataset svhn --n_valid 5000 --labels_per_class 100 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --warmup_iters 1000 --n_steps 40 
```

