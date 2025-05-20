# 学生课堂行为识别系统

## 项目概述

本项目旨在通过YOLOv8模型对学生课堂行为进行识别和分析。系统包含一个GUI应用 (`show.py`)，用于实时/视频文件中的行为检测和统计数据可视化，以及支持训练自定义YOLOv8模型的流程。

## 主要功能 (GUI: `show.py`)

*   **视频上传与播放**: 支持上传本地视频文件，并进行播放、暂停、停止控制。
*   **行为识别**: 对视频中的学生行为进行检测（例如：举手、读书、写字、用手机、趴在桌上/低头）。
*   **人物检测**: 结合通用人物检测模型，辅助识别未有特定行为的学生为"正常"状态。
*   **实时统计与可视化**: 显示检测到的行为统计数据，并提供条形图和饼图进行可视化。
*   **控制台输出**: 显示详细的日志信息和检测结果。

## 项目结构 (推荐)

```
.
├── model_data/
│   ├── best.pt         # 自定义训练的行为识别模型
│   ├── yolov8n.pt      # (可选) YOLOv8n 预训练模型 (用于训练自定义模型的基础权重)
│   └── yolov8l.pt      # YOLOv8l 预训练模型 (用于GUI中的人物检测)
├── datasets/           # (训练时需要)
│   ├── images/
│   │   ├── train/      # 训练图片
│   │   └── val/        # 验证图片
│   ├── labels/
│   │   ├── train/      # 训练标签
│   │   └── val/        # 验证标签
│   └── datasets.yaml   # 数据集配置文件
├── runs/               # YOLOv8 训练输出目录
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt # 训练生成的最佳权重
│               └── last.pt # 最后一次的权重
├── show.py        # GUI 应用主程序
├── README.md           # 本文件
└── requirements.txt    # Python 依赖项
```

## 安装与设置

### 1. 先决条件

*   Python 3.8+
*   pip

### 2. 克隆仓库 

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 下载预训练模型

*   **行为识别基础模型 (可选, 用于训练)**: 下载 `yolov8n.pt` (或其他大小的YOLOv8预训练模型，如 `yolov8s.pt` 等) 从 [YOLOv8 Releases](https://github.com/ultralytics/assets/releases)。将其放入 `model_data/` 目录 (如果计划从头训练或使用特定基础权重进行微调)。训练脚本中会指定模型路径。
*   **人物检测模型 (GUI需要)**: 下载 `yolov8l.pt` 从 [YOLOv8 Releases](https://github.com/ultralytics/assets/releases)。将其放入 `model_data/` 目录。`show.py` 使用此模型进行通用人物检测。
*   **自定义训练模型**: 训练完成后，将生成的 `best.pt` 文件从 `runs/detect/train/weights/best.pt` 复制到 `model_data/best.pt` 供GUI使用。

## 自定义 YOLOv8 模型训练 

### 1. 数据集准备

*   **目录结构**:
    *   `datasets/images/train/`: 存放训练图片。
    *   `datasets/images/val/`: 存放验证图片。
    *   `datasets/labels/train/`: 存放训练标签 (YOLO格式的 `.txt` 文件，与图片名对应)。
    *   `datasets/labels/val/`: 存放验证标签。
*   **`datasets.yaml` 文件**: 在 `datasets/` 目录下创建 `datasets.yaml` 文件，内容如下 (根据您的类别进行调整)：

    ```yaml
    path: ./datasets  # 数据集根目录 (相对于项目根目录或使用绝对路径)
    train: images/train  
    val: images/val  
    
    # 类别数量
    nc: 5
    
    # 类别名称 (顺序必须与标注时的类别ID对应, 0 到 nc-1)
    names: ['hand-raising', 'reading', 'writing', 'using phone', 'learning over the table']
    ```

### 2. 训练流程

在项目根目录下运行以下命令开始训练：
```bash
yolo detect mode=train model=yolov8n.pt data=datasets/datasets.yaml 
```
*   `model`: 使用的预训练模型权重 (例如 `yolov8n.pt`)。
*   `data`: 指向 `datasets.yaml` 文件的路径。

训练完成后，最佳模型权重将保存在 `runs/detect/train/weights/best.pt`。

### 3. 使用训练好的模型

将训练得到的 `runs/detect/train/weights/best.pt` 复制到 `model_data/best.pt`，以便 `show.py` 使用。

## 运行 GUI 应用 (`show.py`)

1.  确保 `model_data/best.pt` (训练的模型) 和 `model_data/yolov8l.pt` (通用人物检测模型) 已放置好。
2.  在项目根目录下运行：
    ```bash
    python show.py
    ```
3.  **GUI 使用**:
    *   通过菜单 "视频源" -> "上传视频" 来加载视频。
    *   使用 "播放/暂停", "停止" 按钮控制视频。
    *   通过菜单 "视频处理" -> "开始行为识别" 来启动/停止检测。
    *   通过 "视频处理" -> "显示行为统计" 查看图表。

## 注意事项
*   **GPU**: 为了获得较好的训练和推理速度，建议使用支持 CUDA 的 NVIDIA GPU。如果未检测到 GPU，程序将默认使用 CPU。
*   **模型路径**: 请仔细检查所有脚本中引用的模型路径是否正确。
*   **依赖版本**: `requirements.txt` 中列出的版本是建议版本，根据您的环境可能需要调整。 