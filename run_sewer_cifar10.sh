bash train_nn_ebm.sh \
    --lr .002 \
    --use_cnn \
    --batch_size 100 \
    --dataset cifar10 \
    --n_valid 5000 \
    --labels_per_class 100 \
    --optimizer adam \
    --p_x_weight 0.0 \
    --p_y_given_x_weight 1.0 \
    --p_x_y_weight 0.0 \
    --sewer_p_y_given_x_weight 1.0 \
    --sigma .03 \
    --warmup_iters 1000 \
    --n_steps 40 \
    --save_dir ./experiment/cifar10_sewer_test1 \
    --plot_uncond \
    --print_every 1 \
    --swish

    # --sm_dim 16 \
    # --l2_energy_reg 0.001 \
    # --l2_energy_reg_neg \
    # --sgld_rmsp 1e-2 \
    # conv1_semisup_sm16_swish_l2sss \
