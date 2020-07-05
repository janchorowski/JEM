# JEM (Joint Energy-Based Models) for Semi-Supervised Classification

Code for the paper "Joint Energy-Based Models for Semi-Supervised Classification". 

This work builds upon [the original JEM work](https://github.com/wgrathwohl/JEM).

Many thanks to my amazing co-authors: [Jörn-Henrick Jacobsen](https://jhjacobsen.github.io/) and [Will Grathwohl](http://www.cs.toronto.edu/~wgrathwohl/). 


## Experiments
### Toy Datasets
To train and evaluate JEM on the rings (concentric circles) dataset:
```markdown
python train_nn_ebm.py --labels_per_class 2 --rings_noise 0.03 --l2_energy_reg 0.0002 --l2_energy_reg_neg --n_rings_data 1000 --lr .001 --use_nn --batch_size 20 --dataset rings --n_valid 200 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --data_seed 20 --plot_uncond --warmup_iters 10 --save_dir . --weight_decay .0005 --sgld_lr .00125 --sgld_std .05 --temper_init 1. --ul_batch_size 100 --viz_every 10
```
Use ```--dataset moons``` instead for moons dataset. Use ```--vat``` flag to run VAT. Set ```--p_x_weight 0.0``` to run just the baseline classifier (with ```--dropout --batch_norm``` for regularization).

### Evaluation

To evaluate the classifier (on CIFAR10):
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test
```
To do OOD detection (on CIFAR100)
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval OOD --ood_dataset cifar_100
```
To generate a histogram of OOD scores like Table 2
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn --save_dir /YOUR/HIST/FOLDER
```
To generate new unconditional samples
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval uncond_samples --save_dir /YOUR/SAVE/DIR --n_sample_steps {THE_MORE_THE_BETTER (1000 minimum)} --buffer_size 10000 --n_steps 40 --print_every 100 --reinit_freq 0.05
```
To generate conditional samples from a saved replay buffer
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --save_dir /YOUR/SAVE/DIR
```
To generate new conditional samples
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --save_dir /YOUR/SAVE/DIR --n_sample_steps {THE_MORE_THE_BETTER (1000 minimum)} --buffer_size 10000 --n_steps 40 --print_every 10 --reinit_freq 0.05 --fresh_samples
 ```


### Attacks

To run Linf attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /PATH/TO/YOUR/MODEL.pt --exp_name /YOUR/EXP/NAME --n_steps_refine 1 --distance Linf --random_init --n_dup_chains 5 --base_dir /PATH/TO/YOUR/EXPERIMENTS/DIRECTORY
```
To run L2 attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /cloud_storage/BEST_EBM.pt --exp_name rerun_ebm_1_step_5_dup_l2_no_sigma_REDO --n_steps_refine 1 --distance L2 --random_init --n_dup_chains 5 --sigma 0.0 --base_dir /cloud_storage/adv_results &
 ```
 

Happy Energy-Based Modeling! 
