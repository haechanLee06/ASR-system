import os
import torch
# 1. 只需要导入 AutoModel，不需要导入 Trainer
from funasr.auto.auto_model import AutoModel

# ================= 配置区域 =================
MODEL_DIR = "/18t/data/home/panyq/models/paraformer-large"
TRAIN_DATA = "/18t/data/home/panyq/llm_data/dataset/funasr_finetune/train"
VAL_DATA = "/18t/data/home/panyq/llm_data/dataset/funasr_finetune/val"
OUTPUT_DIR = "exp/paraformer_ft"
GPU_ID = "0"
# ===========================================

def main():
    # 设置显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print(f"🚀 加载预训练模型: {MODEL_DIR}")
    
    # 2. 加载模型 (AutoModel 会自动读取 config.yaml，解决尺寸匹配问题)
    model = AutoModel(
        model=MODEL_DIR,
        model_revision="v2.0.4",
        disable_update=True,
    )

    # 获取底层 PyTorch 模型
    asr_model = model.model

    # 3. 冻结参数逻辑 (保留你的思路，但针对 Large 模型做了优化)
    print("❄️  正在冻结参数...")
    # 注意：Paraformer-Large 有 50 层 Encoder (encoders.0 ~ encoders.49)
    # 如果你只写 "encoders.10"，中间的 12-49 层会被冻结，这可能不是你想要的。
    # 下面逻辑改为：冻结前 40 层，训练最后 10 层 + Decoder
    
    for name, param in asr_model.named_parameters():
        param.requires_grad = False # 先默认全冻结
        
        # 解冻 Decoder (必须训练)
        if "decoder" in name:
            param.requires_grad = True
        # 解冻 Encoder 的后半部分 (比如 40-49 层)
        # 只要名字里包含 "encoders.4" (40-49) 就解冻
        elif "encoder.encoders.4" in name: 
            param.requires_grad = True
        # 解冻 预测头
        elif "predictor" in name:
             param.requires_grad = True

    # 打印一下验证
    trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    print(f"✅ 参数设置完毕，待训练参数量: {trainable_params}")

    # 4. 开始训练 (这才是正确的入口！)
    # model.train() 会自动构建 DataLoader, Optimizer, Scheduler 并启动 Trainer
    print("🔥 开始训练...")
    model.train(
        model=MODEL_DIR, 
        output_dir=OUTPUT_DIR,
        
        # 传入数据路径列表
        train_data_set_list=[TRAIN_DATA],
        valid_data_set_list=[VAL_DATA],
        
        # 数据读取配置 (A100 暴力配置)
        dataset_conf={
            "index_ds": False,
            "batch_conf": {
                "batch_type": "token",
                "batch_bins": 20000, 
            }
        },
        
        # 训练超参数 (对应你原来 Trainer 里的参数)
        max_epoch=3,
        optim="adam",
        lr=1e-4,
        scheduler="warmuplr",
        
        # 杂项
        log_interval=10,
        use_tensorboard=True,
        val_scheduler_interval=1,
    )

if __name__ == "__main__":
    main()