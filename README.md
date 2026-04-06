# Drone Video Tracking

## Preface

1.无人机目标检测。其实是想要做无人机视频目标跟踪实验，因为目前还是在做最简单的tracking by detection范式，先检测，再预测轨迹，最后匹配跟踪。tracking by detection范式也就是先做检测器，之后再写跟踪器的代码。所以该项目只是完成了第一部分（检测器部分），也就想相当于不是在跟踪了，就只是以一个检测器代码。

2.本次实验没有使用预训练权重，直接修改了yolov11的网络结构，之后所有的权重全是随机初始化的。

3.先打个预防针，该项目本次实验结果很垃圾，mAP50才23%左右。只能说可以看看代码学习一下如何修改yolo模型吧。因为代码里的注释写得很丰富（肯定是ai写的呀）。

## 环境

使用GPU 4090 24G显存训练模型
数据集VisDrone

文件结构:

```
tracking/
│
├── core/                   # 核心训练与推理控制
│   ├── __init__.py
│   └── trainer.py          # 封装好的训练类
│
├── models/                 # 模型架构与自定义网络层
│   ├── __init__.py
│   ├── custom_blocks.py    # 手写的注意力机制模块
│   └── yolov11-p2-gca.yaml # 魔改后的网络结构图
│
├── utils/                  # 工具类（写 NWD Loss 放在这里）
│   |__ __init__.py
│   |__ loss.py            #  NWD Loss
|
├── data_predeal/               # 存放你的 VisDrone 数据集和转换数据格式
|    |—— yolo_dataset/          # VisDrone转换成YOLO格式的数据集
|    |    |——train/             #训练数据集
|    |    |   |——images/
|    |    |   |——labels/
|    |    |——val/               #验证数据集
|    |        |——images/
|    |        |——labels/
|    |
|    |——data_convert.py         #把VisDrone官方格式转为YOLO格式，数据处理
|    |
│    |——visdrone.yaml           #数据集配置文件
|    
└── main.py                 # 项目入口点
```



对了，训练完成后会得到一个runs文件夹，里面就有结果文件。在..\runs\detect\UAV_Research\yolov11_p2_gca_nwd下。

## 原理

无人机视角最大的特点就是目标极小（甚至只有 10x10 像素），且背景极其复杂。YOLOv11 默认的设计是为了应对日常图片（比如人站在面前），对极小目标的特征提取不够敏感。

- **魔改思路 1：引入 P2 超高分辨率检测头（小目标专武）**
  - **原理**：YOLOv11 默认在 P3、P4、P5 三个尺度上进行检测（即下采样 8倍、16倍、32倍）。对于无人机航拍，32倍下采样后，小车早就缩成 1 个像素，甚至消失了！
  - **做法**：修改 YOLOv11 的 yaml 结构，增加一层 **P2 层（下采样 4 倍）**的特征融合与检测头。这样网络能看到更清晰的底层细节，极大地提升小目标召回率。
- **魔改思路 2：引入全局上下文注意力（解决误检）**
  - **原理**：在鄙人使用官方未魔改的yolov8s模型在VisDrone数据集上进行微调时，得到的结果井盖被认成车，可能是因为网络只盯着那个“黑色方块”看。如果我们扩大它的“感受野”，让网络注意到“这个黑块紧贴着地面，没有立体阴影”，它就不会认错了。
  - **做法**：在 YOLOv11 的 Neck 部分，替换或改进原有的 C2PSA，引入具有大感受野的注意力机制（例如 **Deformable Attention（可变形注意力）** 或 **CAS-ViT**）。让网络在判断时，能结合周围的马路特征一起做决策。
- **魔改思路 3：替换损失函数（NWD Loss）**
  - **原理**：普通的 IoU Loss 对小目标极其不友好（两个极小的框只要错开 1 个像素，IoU 就直接变成 0，导致梯度爆炸或无法收敛）。
  - **做法**：将边界框损失替换为专门针对小目标的 **NWD (Normalized Wasserstein Distance) Loss**，把目标框看作二维高斯分布来计算距离，这在无人机论文中是非常好用的提分点。



## 训练步骤

```bash
#先下载VisDrone2019-MOT数据集
cd data_predeal
#安装依赖
pip install -r requirements.txt
#转换数据格式
python data_convert.py

cd ..
#直接开始训练
python main.py
```

