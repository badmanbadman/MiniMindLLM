基于pytorch的现代大语言模型底层架构解析

学习研究使用

<!-- 
启动web UI
在scripts目录下执行
 -->
streamlit run web_demo.py

1、预训练（学知识）

python train_pretrain.py

监督微调（学对话方式）

python train_full_sft.py

知识蒸馏 (Knowledge Distillation, KD)

注意需要更改train_full_sft.py数据集路径，以及max_seq_len  
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py

 LoRA (Low-Rank Adaptation)
torchrun --nproc_per_node 1 train_lora.py
# or
python train_lora.py