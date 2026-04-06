# main.py
import os
import torch
from ultralytics import YOLO
#from ultralytics.nn.tasks import parse_model

# 导入我们自己手写的模块
from models.custom_blocks import GlobalContextAttention
from utils.loss import inject_nwd_loss
from core.trainer import UAVTrainer

def inject_custom_modules():
    """
    魔法操作强化版：将自定义模块注册到 Ultralytics 引擎的核心命名空间中。
    解决 KeyError 问题，让 parse_model(globals()[m]) 能够顺利找到我们的手写模块。
    """
    # 导入官方的核心任务调度模块
    from ultralytics.nn import tasks 
    from ultralytics.nn import modules
    
    # 将自定义模块强行挂载到 tasks.py 的全局命名空间中（这就是“外挂”的核心精髓）
    setattr(tasks, 'GlobalContextAttention', GlobalContextAttention)
    
    # 为了保险起见，同步挂载到 modules 里
    setattr(modules, 'GlobalContextAttention', GlobalContextAttention)
    
    print("[INFO] 成功将自定义网络层挂载到 Ultralytics 解析引擎中: GlobalContextAttention")


    
def main():
    # 1. 初始化环境变量与自定义模块
    inject_custom_modules()
    inject_nwd_loss()  # 同时注入我们魔改的 NWD Loss，确保训练时能够使用它
    # 检查 4090 是否就绪
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 正在使用计算设备: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] 显卡型号: {torch.cuda.get_device_name(0)}")

    # 2. 构建魔改版的 YOLOv11 模型
    yaml_path = os.path.abspath("models/yolov11-p2-gca.yaml")
    print(f"[INFO] 正在基于 {yaml_path} 构建网络拓扑...")
    
    # 注意：这里我们只传了 yaml，不传 .pt，意味着我们将从零初始化这个新结构的权重
    #而且不仅新结构的权重是随机初始化，原先未改变的结构的权重也是随机初始化的
    #所以数据集太少的话，训练得到的模型很垃圾，只是用VisDrone数据集的话可能不太行
    model = YOLO(yaml_path)
    
    print("[INFO] 模型构建成功！P2 头与 GCA 注意力已就位！")
    
    # 打印一下模型的信息，你能看到我们加进去的层
    model.info()#传入这个参数“detailed=True”能够打印出模型的全部网络结构
    
    # 3. 数据集配置文件准备
    data_yaml_path = "/root/private_data/tracking/data_predeal/visdrone.yaml"

    # 4. 创建训练器实例并启动训练
    trainer = UAVTrainer(
        model=model,
        data_yaml=data_yaml_path,
        project_name="UAV_Research",
        run_name="yolov11_p2_gca_nwd"
    )

    trainer.start_training()


if __name__ == "__main__":
    main()