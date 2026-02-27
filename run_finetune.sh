#!/bin/bash

# ==========================================
# 1. 显卡设置 (锁定 GPU 1)
# ==========================================
export CUDA_VISIBLE_DEVICES="1"
gpu_num=1

# ==========================================
# 2. 路径设置
# ==========================================
train_tool="/18t/data/home/panyq/miniconda3/envs/asr_ft/lib/python3.9/site-packages/funasr/bin/train.py"
model_dir="/18t/data/home/panyq/models/paraformer-large" 
train_data="/18t/data/home/panyq/llm_data/dataset/funasr_finetune/train.jsonl"
val_data="/18t/data/home/panyq/llm_data/dataset/funasr_finetune/val.jsonl"
output_dir="/18t/data/home/panyq/llm_output/paraformer_sichuan_lora" 
mkdir -p ${output_dir}

# ==========================================
# 3. 启动 LoRA 训练
# ==========================================
echo "🚀 正在启动 Paraformer LoRA 微调..."
echo "✅ 锁定 GPU 1 | 显式指定模型参数 | Batch Size: 5000"

torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_num} \
    "${train_tool}" \
    \
    ++model="BiCifParaformer" \
    ++init_param="${model_dir}/model.pt" \
    ++tokenizer_conf.token_list="${model_dir}/tokens.json" \
    ++tokenizer_conf.seg_dict_file="${model_dir}/seg_dict" \
    ++frontend_conf.cmvn_file="${model_dir}/am.mvn" \
    \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset="AudioDataset" \
    ++dataset_conf.index_ds="IndexDSJsonl" \
    ++dataset_conf.batch_sampler="BatchSampler" \
    ++dataset_conf.batch_type="token" \
    ++dataset_conf.batch_size=5000 \
    ++dataset_conf.max_token_length=2000 \
    ++dataset_conf.num_workers=4 \
    \
    ++train_conf.max_epoch=10 \
    ++train_conf.log_interval=10 \
    ++train_conf.resume=true \
    ++train_conf.validate_interval=2000 \
    ++train_conf.save_checkpoint_interval=2000 \
    ++train_conf.ignore_init_mismatch=true \
    \
    ++optim_conf.lr=0.0005 \
    ++output_dir="${output_dir}" \
    \
    ++train_conf.freeze_param=true \
    ++model_conf.use_lora=true \
    ++encoder="SANMEncoder" \
    ++encoder_conf='{"output_size":512}' \
    ++model_conf.lora_rank=8 \
    ++model_conf.lora_alpha=32 \
    ++model_conf.lora_dropout=0.1 \
    ++model_conf.lora_list='["encoder", "decoder"]'