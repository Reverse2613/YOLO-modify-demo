# core/trainer.py
import os

class UAVTrainer:
    """
    无人机目标跟踪的检测模型训练器 (UAV Tracking Model Trainer)
    将复杂的训练超参数封装在类中，方便后续做消融实验和版本管理。
    """
    def __init__(self, model, data_yaml, project_name="UAV_Research", run_name="yolov11_p2_gca_nwd"):
        """
        初始化训练器
        :param model: 已经构建好并注入了魔法（GCA+NWD）的 YOLO 模型实例
        :param data_yaml: 数据集配置文件的绝对或相对路径
        :param project_name: 整个实验的大文件夹名
        :param run_name: 本次特定实验的子文件夹名 (比如我们用了 p2, gca, nwd，就起这个名字)
        """
        self.model = model
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.run_name = run_name

    def start_training(self):
        """
        启动训练流程
        这里的参数是专为 RTX 4090 (24G) 和 VisDrone 无人机数据集量身定制的。
        """
        print(f"\n[启动训练] 准备开始训练项目: {self.project_name}/{self.run_name}")
        print(f"[数据集路径]: {self.data_yaml}")
        
        # 调用模型的 train 方法，这里面包含了 Ultralytics 底层所有的循环
        results = self.model.train(
            data=self.data_yaml,
            
            # ================= 1. 核心训练时长 =================
            epochs=300,        # 论文级别通常需要 100-300 轮。因为我们用 4090，100 轮很快，很快个蛋。
            patience=50,       # 早停机制(Early Stopping)：如果连续 50 轮 mAP 没有提升，自动停止，防止过拟合。
            
            # ================= 2. 硬件与性能榨取 =================
            batch=16,          # 4090 24G显存，输入640图片，batch=16比较安全且高效 (如果有余力可以尝试 32，但多半不行，我试过了)
            imgsz=640,         # 输入网络前将图片缩放到 640x640 (由于我们加了 P2，对于小目标已经很友好了)
            device="0",        # 指定使用第一块 GPU (4090)
            workers=8,         # 开启 8 个 CPU 线程负责将图片从硬盘搬运到显存
            amp=True,          # 【开启自动混合精度】极大提速并节省显存！
            cache=True,        # 【开启内存缓存】如果你的计算平台服务器内存(RAM)大于32G，开启这个会将数据集常驻内存，免去读盘开销。
            
            # ================= 3. 学习率与优化器策略 =================
            optimizer='AdamW', # 相比传统的 SGD，AdamW 对复杂网络(尤其是带注意力机制的网络)收敛更好。
            lr0=0.001,         # 初始学习率
            cos_lr=True,       # 【开启余弦退火学习率】平滑收敛，涨点神器。
            
            # ================= 4. 保存与输出设置 =================
            project=self.project_name, 
            name=self.run_name,
            save=True,         # 保存训练出最好的权重 (best.pt)
            plots=True,        # 自动画出各种损失函数曲线和 PR 曲线图（写论文必备！）
        )
        
        print("\n[训练彻底完成] 请查看生成的曲线图与 best.pt 权重文件！")
        return results