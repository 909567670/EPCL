# sdp 持续学习

## 环境 
使用`sdp.yaml`安装conda环境

## 训练
修改并运行脚本 `train_mydatasets_s_dualprompt.sh`
> 目前只实现了实验的训练方式，在已有模型的基础上训练还未完成 `my_train.sh`

## 评估
修改并运行脚本 `my_eval.sh`

## ONNX
导出：`test_export_onnx.py`，得到vit和sdp模型
测试：`test_use_onnx.py`


