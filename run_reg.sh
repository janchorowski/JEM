#!/usr/bin/env bash

BASE="/scratch/gobi1/gwohl/REGRESSION"



for DN in 0. .01 .1
do
    echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset protein \
          --save_dir ${BASE}/protein/REG/dn_${DN} \
          --lr .001 --data_noise ${DN} --weight_decay .0005 --batch_size 128 \
          --epochs 50

    echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset power_plant \
          --save_dir ${BASE}/power/REG/dn_${DN} \
          --lr .001 --data_noise ${DN} --weight_decay .0005 --batch_size 128 \
          --epochs 100

    echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset concrete \
          --save_dir ${BASE}/concrete/REG/dn_${DN} \
          --lr .001 --data_noise ${DN} --weight_decay .0005 --batch_size 128 \
          --epochs 250


    for PC in 0. .01 .1 1. 10.
    do
        # PROTEIN
        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset protein \
          --save_dir ${BASE}/protein/EBM/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebm --weight_decay .0005 --batch_size 128 \
          --epochs 50

        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset protein \
          --save_dir ${BASE}/protein/EBR/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebr --weight_decay .0005 --batch_size 128 \
          --epochs 50


        # POWER
        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset power_plant \
          --save_dir ${BASE}/power/EBM/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebm --weight_decay .0005 --batch_size 128 \
          --epochs 100

        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset power_plant \
          --save_dir ${BASE}/power/EBR/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebr --weight_decay .0005 --batch_size 128 \
          --epochs 100


        # CONCRETE
        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset concrete \
          --save_dir ${BASE}/concrete/EBM/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebm --weight_decay .0005 --batch_size 128 \
          --epochs 250

        echo srun -c 2 --mem=10G -p cpu\
        python regression.py --dataset concrete \
          --save_dir ${BASE}/concrete/EBR/dn_${DN}_pc_${PC} \
          --loss ml --p_control ${PC} \
          --lr .001 --data_noise ${DN} --mcmc_steps 25 --ebr --weight_decay .0005 --batch_size 128 \
          --epochs 250
    done
done
