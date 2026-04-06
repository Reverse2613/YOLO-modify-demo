import os
import cv2
from pathlib import Path

# VisDrone MOT 的类别映射 (我们挑出需要识别的常见目标)
#如果不明白为什么数字对应哪个类别，查看官方VisDrone2019-MOT数据集的格式，但是就实验所用的VisDrone2019-MOT数据集，明确就是以下对应关系。
# 原始类别: 0:ignored, 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van, 6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor, 11:others
# 为了简化和提高准确率，我们把它们映射为 YOLO 需要的 0-9 编号，并丢弃不需要的类别(0和11)
VISDRONE_CLASSES = {
    1: 0,  # pedestrian (行人)
    2: 1,  # people (人群)
    3: 2,  # bicycle (自行车)
    4: 3,  # car (小汽车)
    5: 4,  # van (面包车)
    6: 5,  # truck (卡车)
    7: 6,  # tricycle (三轮车)
    8: 7,  # awning-tricycle (带篷三轮车)
    9: 8,  # bus (公交车)
    10: 9  # motor (摩托车)
}

def convert_mot_to_yolo(visdrone_dir, output_dir):
    """
    核心逻辑：遍历每一个视频序列，将 VisDrone 的标注转换为 YOLO 的标注格式。
    """
    visdrone_path = Path(visdrone_dir)
    output_path = Path(output_dir)
    
    # YOLO 要求的目录结构
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seqs_path = visdrone_path / 'sequences'    # 存放图片序列的文件夹
    annos_path = visdrone_path / 'annotations' # 存放 txt 标注的文件夹

    if not seqs_path.exists() or not annos_path.exists():
        print(f"错误：找不到 {seqs_path} 或 {annos_path}，请检查路径！")
        return

    # 遍历每一个视频序列 (比如 uav0000013_00000_v)
    for seq_name in os.listdir(seqs_path):
        seq_dir = seqs_path / seq_name
        anno_file = annos_path / f"{seq_name}.txt"
        
        if not anno_file.exists():
            continue
            
        print(f"正在处理视频流序列: {seq_name} ...")

        # 步骤 1：读取该视频流的完整标注数据
        with open(anno_file, 'r') as f:
            lines = f.readlines()

        # 建立一个字典，把属于同一帧的标注归拢到一起
        frame_annotations = {}
        for line in lines:
            # VisDrone 格式: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            parts = line.strip().split(',')
            frame_idx = int(parts[0])
            category = int(parts[7])
            
            # 过滤掉不需要的类别（如 ignored 或 others）
            if category not in VISDRONE_CLASSES:
                continue
                
            yolo_class = VISDRONE_CLASSES[category]
            # 提取边界框坐标
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append((yolo_class, bbox_left, bbox_top, bbox_width, bbox_height))

        # 步骤 2：处理这个视频流序列中的每一张图片
        for img_name in os.listdir(seq_dir):
            if not img_name.endswith('.jpg'):
                continue
                
            # 从图片名提取帧号 (比如 0000001.jpg -> 1)
            frame_idx = int(img_name.split('.')[0])
            img_path = seq_dir / img_name
            
            # 如果这一帧没有我们需要的目标，就跳过
            if frame_idx not in frame_annotations:
                continue

            # 读取一张图片只是为了获取它的宽和高 (用于归一化计算)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # 步骤 3：计算 YOLO 格式并写入新的 txt 文件
            # 目标文件名，例如：uav0000013_00000_v_0000001.jpg
            new_img_name = f"{seq_name}_{img_name}"
            new_label_name = new_img_name.replace('.jpg', '.txt')
            
            label_content =[]
            for (cls_id, x_left, y_top, box_w, box_h) in frame_annotations[frame_idx]:
                # 算法原理：将左上角坐标转换为中心点坐标，并除以图像总宽高进行归一化(变成0~1之间的小数)
                x_center = (x_left + box_w / 2) / img_w
                y_center = (y_top + box_h / 2) / img_h
                norm_w = box_w / img_w
                norm_h = box_h / img_h
                
                # 确保坐标在 0-1 之间，防止越界
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                norm_w, norm_h = max(0, min(1, norm_w)), max(0, min(1, norm_h))
                
                label_content.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            if label_content:
                # 把图片做一个软链接或者干脆只记录路径（为了不占用更多的硬盘空间，这里我们直接写标注，后续训练依靠绝对路径或复制软链接）
                # 为了新手友好，我们直接在 output 里创建原图片的软链接（快捷方式），这样不额外占硬盘空间！
                symlink_img = images_dir / new_img_name
                if not symlink_img.exists():
                    os.symlink(img_path.absolute(), symlink_img)
                
                # 写入 YOLO 格式的 txt 标注
                with open(labels_dir / new_label_name, 'w') as f:
                    f.write('\n'.join(label_content))

    print(f"\n{visdrone_dir} 序列转换完成！结果保存在 {output_dir}")

if __name__ == '__main__':
    # ================= 注意 =================
    # 请根据你实际下载的文件夹名称修改这里的路径
    # ========================================
    print("开始转换 Training 数据集...")
    convert_mot_to_yolo('/root/private_data/VisDrone2019-MOT-train', 'yolo_dataset/train')
    
    print("\n开始转换 Validation 数据集...")
    convert_mot_to_yolo('/root/private_data/VisDrone2019-MOT-val', 'yolo_dataset/val')
    
    print("\n恭喜！数据准备完毕，可以开始准备训练了！")