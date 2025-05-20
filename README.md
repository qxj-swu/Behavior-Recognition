# 智慧课堂学情分析系统

## 项目概述

本项目旨在通过YOLOv8模型对学生课堂行为进行识别和分析。系统包含一个GUI应用 (`show_0415.py`)，用于实时/视频文件中的行为检测和统计数据可视化，以及支持训练自定义YOLOv8模型的流程。

## 主要功能 (GUI: `show_0415.py`)

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
├── 学生课堂行为识别.md     # 项目相关文档 (源文档)
├── show_0415.py        # GUI 应用主程序
├── test.py             # (可选) 命令行预测脚本 (源自 `学生课堂行为识别.md`)
├── README.md           # 本文件
└── requirements.txt    # Python 依赖项
```

## 安装与设置

### 1. 先决条件

*   Python 3.8+
*   pip

### 2. 克隆仓库 (示例)

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. 创建虚拟环境 (推荐)

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
*   **人物检测模型 (GUI需要)**: 下载 `yolov8l.pt` 从 [YOLOv8 Releases](https://github.com/ultralytics/assets/releases)。将其放入 `model_data/` 目录。`show_0415.py` 和 `test.py` (源自markdown) 使用此模型进行通用人物检测。
*   **自定义训练模型**: 训练完成后，将生成的 `best.pt` 文件从 `runs/detect/train/weights/best.pt` 复制到 `model_data/best.pt` 供GUI和 `test.py` 使用。

## 自定义 YOLOv8 模型训练 (基于 `学生课堂行为识别.md`)

### 1. 数据集准备

参考 `学生课堂行为识别.md` 中 "4.2 学生课堂行为识别数据集构建" 章节。

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
    **注意**: `学生课堂行为识别.md` 中YAML示例的第五个类别是 `'learning over the table'`。请确保这与您的标注一致。

*   **Ultralytics 设置文件 (datasets_dir)**:
    YOLOv8 需要知道数据集的根目录。根据 `学生课堂行为识别.md`，您可能需要修改 Ultralytics 的设置文件。
    *   **Linux**: 打开 (或创建) `/home/YOUR_USER/.config/Ultralytics/settings.json` (将 `YOUR_USER` 替换为您的用户名)。
    *   **Windows**: 打开 (或创建) `C:\Users\YOUR_USER\AppData\Roaming\Ultralytics\settings.json` (将 `YOUR_USER` 替换为您的用户名)。
    *   修改 (或添加) `"datasets_dir"` 的值为 `datasets` 文件夹所在的路径，或者，如markdown建议，可以设置为 `"./"` (点斜杠)，表示当前工作目录的相对路径，前提是您的 `datasets` 文件夹在项目根目录，并且您从项目根目录运行训练命令。
        ```json
        {
          "datasets_dir": "./", // 或者 "D:/path/to/your/project/datasets"
          // ... 其他设置
        }
        ```
    Alternatively, you can often specify the full path to `datasets.yaml` in the training command, and ensure `path` within `datasets.yaml` is correct (e.g., `path: ../datasets` if `datasets.yaml` is in a subfolder and paths are relative to it, or an absolute path). The markdown's approach of setting `datasets_dir` in global settings is one way to manage this.

### 2. 训练流程

在项目根目录下运行以下命令开始训练 (参考 `学生课堂行为识别.md` 中 "4.3 ...模型训练" 章节的命令)：

```bash
yolo detect mode=train model=yolov8n.pt data=datasets/datasets.yaml epochs=50 batch=16 workers=1 imgsz=160
```

*   `model`: 使用的预训练模型权重 (例如 `yolov8n.pt`)。
*   `data`: 指向 `datasets.yaml` 文件的路径。
*   `epochs`: 训练轮数。
*   `batch`: 批大小。
*   `imgsz`: 输入图像大小。`160` 是markdown中的示例，通常使用 `640` 以获得更好性能，但这需要更多计算资源。

训练完成后，最佳模型权重将保存在 `runs/detect/train/weights/best.pt`。

### 3. 使用训练好的模型

将训练得到的 `runs/detect/train/weights/best.pt` 复制到 `model_data/best.pt`，以便 `show_0415.py` 和 `test.py` 可以使用它。

## 运行 GUI 应用 (`show_0415.py`)

1.  确保 `model_data/best.pt` (您训练的模型) 和 `model_data/yolov8l.pt` (通用人物检测模型) 已放置好。
2.  在项目根目录下运行：

    ```bash
    python show_0415.py
    ```

3.  **GUI 使用**:
    *   通过菜单 "视频源" -> "上传视频" 来加载视频。
    *   使用 "播放/暂停", "停止" 按钮控制视频。
    *   通过菜单 "视频处理" -> "开始行为识别" 来启动/停止检测。
    *   通过 "视频处理" -> "显示行为统计" 查看图表。

4.  **关于类别名称的注意事项**:
    `show_0415.py` 内部定义了行为类别 (`BEHAVIOR_CLASSES`) 和相应的置信度配置 (`default_confidence`, `behavior_adjustment`)。这些是：
    ```python
    BEHAVIOR_CLASSES = [
        'hand-raising',
        'reading',
        'writing',
        'using phone',
        'bowing head', # 注意：这与 datasets.yaml 中的 'learning over the table' 不同
        'normal' # 'normal' 是用于统计和显示的特殊类别
    ]
    default_confidence = {
        'hand-raising': 0.1,
        'reading': 0.1,
        'writing': 0.1,
        'using phone': 0.1,
        'bowing head': 0.05, # 同样，'bowing head'
    }
    ```
    如果您训练的模型使用了 `datasets.yaml` 中定义的5个类别 (特别是 `'learning over the table'`)，您可能需要修改 `show_0415.py` 中的 `BEHAVIOR_CLASSES`, `BEHAVIOR_COLORS`, `default_confidence`, 和 `behavior_adjustment` 以使其与您训练的模型的类别完全匹配，确保正确的行为文本显示和置信度调整。或者，确保您的 `model_data/best.pt` 模型输出的类别名称被 `show_0415.py` 正确处理 (它通过 `behavior_model.names.get(cls_idx)` 获取类别名，这应该来自模型本身)。

## 运行命令行预测脚本 (`test.py` - 源自 `学生课堂行为识别.md`)

`学生课堂行为识别.md` 文件中提供了一个 `test.py` 脚本，用于对单张图片或视频进行预测，并结合了人物检测来识别 "normal" 行为。

1.  **脚本修改**:
    *   确保 `test.py` 中的模型路径正确无误 (应为 `model_data/best.pt` 和 `model_data/yolov8l.pt`)。
        ```python
        # 在 test.py 的 main 函数中:
        behavior_model = YOLO("model_data/best.pt")
        person_model = YOLO('model_data/yolov8l.pt')
        ```
    *   修改 `input_path` 和 `output_dir` 变量为您要测试的媒体文件路径和输出目录。
        ```python
        # 在 test.py 的 main 函数中:
        input_path = "path/to/your/input/image_or_video.mp4" # 修改这里
        output_dir = "path/to/your/output_directory"        # 修改这里
        ```
2.  **运行脚本**:
    在项目根目录下 (假设 `test.py` 也在根目录):
    ```bash
    python test.py
    ```
    处理后的图片或视频将保存在您指定的 `output_dir` 中。

3.  **类别注意事项 (`test.py`)**:
    `test.py` 中定义的 `BEHAVIOR_CLASSES` 是：
    ```python
    BEHAVIOR_CLASSES = [
        'hand-raising',
        'reading',
        'writing',
        'using phone',
        'bowing head',
        'leaning over the table', # 注意：这与 datasets.yaml 中的 'learning over the table' 拼写/用词可能不同
    ]
    ```
    同样，请确保这些类别与您训练的 `model_data/best.pt` 模型所识别的类别一致或按需调整 `test.py`。

## 注意事项

*   **GPU**: 为了获得较好的训练和推理速度，建议使用支持 CUDA 的 NVIDIA GPU。如果未检测到 GPU，程序将默认使用 CPU。
*   **模型路径**: 请仔细检查所有脚本中引用的模型路径是否正确。
*   **`imgsz` 参数**: 在训练时，增大 `imgsz` (如 `640`) 通常能提高模型精度，但会增加显存消耗和训练时间。
*   **依赖版本**: `requirements.txt` 中列出的版本是建议版本，根据您的环境可能需要调整。 