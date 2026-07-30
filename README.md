# 图书馆书脊标签离线识别

基于 RapidOCR（ONNX Runtime）的离线 OCR 方案，识别书脊白色标签上的编号，如 `A03-0068`、`B04-0007`。

## 环境准备

```bash
cd /Users/wujie/Desktop/code/图书馆视觉
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 说明：本方案不依赖 Tesseract，也不调用云端 API。RapidOCR 第一次运行时会自动下载 ONNX 模型到本地缓存。

## 使用方法

### 识别单张图片

```bash
source venv/bin/activate
python library_label_ocr.py 图书馆图片_jpg/21.jpg
```

### 批量识别整个目录并生成可视化图

```bash
source venv/bin/activate
python library_label_ocr.py 图书馆图片_jpg --visualize --output ocr_results.json
```

结果会输出到 `ocr_results.json`，可视化图保存在 `output/` 目录。

## 输出格式

```json
[
  {
    "image": "图书馆图片_jpg/21.jpg",
    "label_count": 19,
    "labels": [
      {"label": "A04-0077", "cx": 1234.5, "cy": 567.8, "raw_text": "A04. + 0077", "score": 0.74},
      ...
    ],
    "elapsed_ms": {"detection": ..., "classification": ..., "recognition": ...},
    "visualization": "output/21_result.jpg"
  }
]
```

## 目录结构

```
图书馆视觉/
├── venv/                       # Python 虚拟环境
├── 图书馆图片/                 # 原始 HEIC 照片
├── 图书馆图片_jpg/             # 转换后的 JPG（脚本读取这里）
├── output/                     # 可视化结果
├── library_label_ocr.py        # 主脚本
├── requirements.txt
├── ocr_results.json            # 最新识别结果
└── README.md
```

## 识别策略

当前脚本采用“粗检测 + 局部放大 + 空间-语义后处理”的多阶段流程：

1. **EXIF 方向校正**：用 PIL 读取并校正照片方向，保证书脊竖直。
2. **整图粗 OCR**：RapidOCR 全图检测所有文字框。
3. **小区域聚类放大**：把距离很近的 OCR 框（通常是同一标签的上下两行）聚类，再对局部做 3 倍放大 + CLAHE 增强后二次识别。
4. **上下行配对**：标签贴纸是“前缀在上、数字在下”的两行文本，按垂直对齐关系配对成完整编号。
5. **空间 NMS + 字符串去重**：对配对结果按包围盒做非极大值抑制，避免一本书被重复识别。
6. **行级一致性校正**：
   - 同一书架层的标签前缀应一致，用多数票修正误读字母（如 `D04` → `B04`）。
   - 同一层编号随 x 坐标单调递增，用 Theil-Sen 稳健拟合修正明显读错的数字（如 `6000` → `0008`）。
7. **远景条带 fallback**：若全图几乎没检出标签，会把图像分成几条水平带，放大 2 倍后再检一次，争取救回远景书架最下面几层的标签。

## 当前效果

| 类型 | 样张 | 检出标签数 | 效果说明 |
|------|------|------------|----------|
| 近景正面（01-07） | 01.jpg | 3/3 | 正确：`B04-0007/0008/0009` |
| 近景正面（02.jpg） | 02.jpg | 4/4 | 正确：`B04-0012~0015` |
| 近景正面（03.jpg） | 03.jpg | 3 | 正确（顺序略有抖动） |
| 近景正面（04.jpg） | 04.jpg | 12 | 基本正确，极个别数字需趋势校正 |
| 近景正面（05-07） | 05-07.jpg | 4/5/8 | 近景基本全对 |
| 近景侧面（11） | 11.jpg | 1/1 | 正确：`A03-0068` |
| 中景（21-24） | 21.jpg | 19 | 效果最好，A04/A03 两层基本全对 |
| 中景（22.jpg） | 22.jpg | 10 | 正确：`B02-0028~0037` |
| 中景（23.jpg） | 23.jpg | 16 | C03/C02 两层基本全对 |
| 中景（24.jpg） | 24.jpg | 8 | 正确：`B03-0021~0029` |
| 远景（31） | 31.jpg | 113 | **能检出大量标签，但数字误读、重复、漏检仍较多**。趋势校正修复了 `6000` 等极端错误，整体可用作“定位+粗盘点”，不建议直接作为精确编号。 |
| 超远景（32、33、35） | 32.jpg | 0 | **当前分辨率下标签低于 OCR 可识别极限**，算法无法恢复。 |
| 远景（34） | 34.jpg | 4 | 能识别出少量较清晰的标签。 |

> 注：`ocr_results.json` 中 `*` 标记表示该标签经历过前缀或数字的趋势校正。

## 限制与下一步建议

1. **31 号这类“远景”仍有误读**：虽然定位到标签位置的成功率已经很高，但数字识别准确率受限于当前照片分辨率。要继续提升，优先让机器人**靠近拍摄**（0.5~1 m）或**使用光学变焦/更高像素相机**。
2. **32、33、35 号“超远景”目前 0 检出**：标签在整图中的像素尺寸已经低于 RapidOCR 检测器的最小响应尺寸，单纯靠软件放大无法无中生有。建议：
   - 机器人移动到书架正前方 0.5 m 以内；
   - 或者摄像头开启 2~3 倍光学变焦后再拍；
   - 若必须远距离拍整面墙，需要更高像素相机（如 8000×6000 以上）。
3. **角度与光照**：尽量让摄像头正对书脊，避免大角度透视；标签是白底黑字，补光能显著提升对比度。
4. **换更强的 OCR 模型**：若算力允许，可尝试 PaddleOCR（PP-OCRv4）或 ChineseOCR，数字识别精度通常更高。
5. **连接机器人摄像头**：把脚本中的文件读取替换为 `cv2.VideoCapture` 即可实时取流；建议先在机器人上复现实验室拍摄距离，再决定最终工作距离。
6. **部署到 Linux/工控机**：当前在 macOS 验证，机器人若为 Linux，安装 `rapidocr-onnxruntime` 同样可用，无需修改代码。

## 真实场景（机器人头部相机）测试

针对 `真实场景图/` 下的 720p 头部相机图片/视频，另外写了一套专门的处理流程。

### 脚本

- `real_scene_ocr.py`：纯 OCR 方案（RapidOCR + 整图放大 + 碎片组合）。
- `yolo_real_scene_ocr.py`：YOLO 切书 + OCR 方案（需要先用 `prepare_yolo_data.py` 准备训练数据）。
- `prepare_yolo_data.py`：根据当前 OCR 成功帧自动生成书的伪标签，供 YOLO 训练。
- `analyze_real_scene_results.py`：汇总 `output/real_scene_results.json` 并打印成功率。
- `vote_real_scene_results.py`：对多帧 OCR 结果做时空投票，输出稳定标签。
- `visualize_voted_labels.py`：在原图上绘制投票后的稳定标签。

### 运行方式

```bash
source venv/bin/activate

# 1) 纯 OCR 方案（当前效果较好）
python real_scene_ocr.py 真实场景图 --output-dir 真实场景图/output --video-step 5 --no-easy

# 2) 多帧投票（视频场景强烈推荐）
python vote_real_scene_results.py
python visualize_voted_labels.py

# 3) YOLO 切书方案（需要更多标注才能稳定）
python prepare_yolo_data.py
# 训练 YOLO（参考 yolo_real_scene_ocr.py 中的训练代码）
python yolo_real_scene_ocr.py 真实场景图/录制视频/head_20260728_160045.mp4 --output-dir 真实场景图/output --conf 0.05
```

### 当前结果（路线 1 优化后）

- **纯 OCR 方案**：
  - 单帧：22 帧视频中识别出 **15 帧**，标签为 `B04-0015`。
  - 多帧投票：把 22 帧的结果聚类投票后，得到 **1 个稳定标签 `B04-0015`**，共 2962 个有效检测参与投票，平均置信度 0.635。投票可视化图见 `output/*_voted.jpg`。
- **YOLO 切书方案**：由于训练数据只有 15 张 OCR 成功帧生成的伪标签，模型只能召回视频右上角这一本书，且置信度很低（~0.08），22 帧中仅识别出 **7 帧**。
- **head_lossless 静态图**：23 张全部失败，720p 下书脊标签像素尺寸过低，超出当前 OCR 可恢复范围。

### 真实场景限制与建议

1. **720p 分辨率是最大瓶颈**：头部相机画面中书脊标签只有约 30~60 像素高，字符仅 5~10 像素，已经接近普通 OCR 的识别极限。要让这套方案稳定工作，建议：
   - 让机器人靠近书架到 0.3~0.5 m 再拍摄；
   - 或使用更高分辨率相机（1080p/2K/4K）/ 光学变焦。
2. **YOLO 需要更多真实标注**：当前伪标签全部来自同一本书的相近位置，模型过拟合严重。如果要走 YOLO 切书路线，需要人工标注 50~100 张不同姿态、距离、光照下的书脊/整书 bbox。
3. **数据增强有用但有限**：放大、锐化、CLAHE 能帮到“刚好在临界分辨率”的帧；对于严重欠采样或运动模糊的超远景帧，软件增强无法恢复丢失的信息。
4. **后续可尝试方向**：
   - 用更高质量的 OCR（PaddleOCR PP-OCRv4、Surya 等）替换 RapidOCR；
   - 对视频做多帧超分辨率/多帧融合，提升信噪比；
   - 训练端到端的“书脊标签检测+识别”专用模型。
