#!/bin/bash

# measure command execution time
measure_time() {
    start_time=$(date +%s.%N)
    eval "$1"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Command '$1' took $elapsed_time seconds"
}

# define checkpoint paths
lba_model_ckpt_path="checkpoints/LBA/model_1_epoch_205_rmse_1_352_pearson_0_612_spearman_0_609.ckpt"
psr_model_ckpt_path="checkpoints/PSR/model_epoch_115_localpearson_0_616_localspearman_0_532_localkendall_0_385_globalpearson_0_871_globalspearman_0_869_globalkendall_0_676.ckpt"
nms_small_model_ckpt_path="checkpoints/NMS/NMS_Small/model_epoch_9977_mse_0_0070.ckpt"
nms_small_20body_model_ckpt_path="checkpoints/NMS/NMS_Small_20Body/model_epoch_10087_mse_0_0071.ckpt"
nms_static_model_ckpt_path="checkpoints/NMS/NMS_Static/model_epoch_5159_mse_0_0073.ckpt"
nms_dynamic_model_ckpt_path="checkpoints/NMS/NMS_Dynamic/model_epoch_9825_mse_0_0173.ckpt"
rs_model_ckpt_path="checkpoints/RS/model_1_epoch_54_accuracy_0_9873.ckpt"

# measure time taken for for LBA task
measure_time "python3 src/eval.py datamodule=atom3d_lba model=gcpnet_lba logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$lba_model_ckpt_path\""

# measure time taken for for PSR task
measure_time "python3 src/eval.py datamodule=atom3d_psr model=gcpnet_psr logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$psr_model_ckpt_path\""

# measure time taken for for NMS tasks
measure_time "python3 src/eval.py datamodule=nms datamodule.data_mode=small model=gcpnet_nms logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$nms_small_model_ckpt_path\""
measure_time "python3 src/eval.py datamodule=nms datamodule.data_mode=small_20body model=gcpnet_nms logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$nms_small_20body_model_ckpt_path\""
measure_time "python3 src/eval.py datamodule=nms datamodule.data_mode=static model=gcpnet_nms logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$nms_static_model_ckpt_path\""
measure_time "python3 src/eval.py datamodule=nms datamodule.data_mode=dynamic model=gcpnet_nms logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$nms_dynamic_model_ckpt_path\""

# measure time taken for for RS task
measure_time "python3 src/eval.py datamodule=rs model=gcpnet_rs logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path=\"$rs_model_ckpt_path\""
