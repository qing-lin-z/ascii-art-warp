# ASCII Art Warp

将视频转换为彩色 ASCII 艺术风格，支持 GPU 加速。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 效果展示

将任意视频实时转换为 ASCII 字符艺术，支持彩色输出和终端播放。

```
输入: 640x360 彩色视频
输出: 80x45 ASCII 字符帧（可自定义分辨率）
```

## 功能特性

- **双模式运行** — 图形界面 (GUI) 和命令行 (CLI) 双模式
- **GPU 加速** — PyTorch CUDA 加速渲染（自动降级到 NumPy）
- **彩色输出** — 保留原始视频色彩，字符间隙显示黑色背景
- **终端播放** — `--watch` 实时终端播放模式（支持彩色 ANSI）
- **视频导出** — 输出 MP4 视频文件（含 H.264 压缩）
- **可自定义** — 字体、字符集、分辨率、背景色、曝光度均可调节
- **跨平台** — Windows / macOS / Linux

## 快速开始

### Windows 一键安装（推荐）

双击 `install.bat`，脚本自动完成：

1. 检测 Python 环境
2. 安装核心依赖（numpy, opencv, pillow, pygame）
3. 检测 NVIDIA 显卡，自动选择 CUDA 或 CPU 版 PyTorch
4. 安装可选加速库（numba, moviepy）
5. 检测 ffmpeg（视频编码必需）
6. 验证所有组件

安装完成后，双击 `ascii_art_warp_final.py` 即可启动 GUI，或将视频文件拖到程序上自动处理。

### macOS / Linux 手动安装

```bash
# 1. 安装依赖
pip install numpy opencv-python pillow pygame
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # GPU 加速（如有 NVIDIA 显卡）

# 2. 安装 ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 3. 运行（GUI 模式）
python ascii_art_warp_final.py
```

## 使用指南

### GUI 模式

双击 `ascii_art_warp_final.py` 或直接运行，打开图形界面：

1. **选择视频** — 点击「选择视频文件」
2. **调整参数** — 分辨率、字体、字符集、背景色、曝光度
3. **预览** — 点击「预览」查看实时效果
4. **导出** — 点击「导出视频」生成 MP4

### CLI 模式

```bash
# 基本用法
python ascii_art_warp_final.py --cli -i input.mp4 -o output.mp4

# 指定分辨率
python ascii_art_warp_final.py --cli -i input.mp4 -o output.mp4 --cols 80 --rows 45

# 彩色输出
python ascii_art_warp_final.py --cli -i input.mp4 -o output.mp4 --color

# 调整曝光度（1.0=原始，1.6=推荐）
python ascii_art_warp_final.py --cli -i input.mp4 -o output.mp4 --color --exposure 1.6
```

### 终端播放模式（--watch）

```bash
# 在终端中实时播放 ASCII 艺术视频
python ascii_art_warp_final.py --watch -i input.mp4 --color
```

### 全部参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cli` | 命令行模式（无 GUI） | GUI 模式 |
| `--watch` | 终端实时播放模式 | 关闭 |
| `-i` | 输入视频文件路径 | 必填 |
| `-o` | 输出视频文件路径 | output.mp4 |
| `--cols` | ASCII 字符列数 | 80 |
| `--rows` | ASCII 字符行数 | 45 |
| `--color` | 彩色输出 | 灰度 |
| `--charset` | 字符集 | blocky |
| `--font` | 字体路径 | Consolas |
| `--exposure` | 曝光度倍率 (1.0-3.0) | 1.6 |
| `--bg-color` | 背景色 (十六进制) | #000000 |
| `--bg-r` `--bg-g` `--bg-b` | 背景色 RGB 分量 | 0 0 0 |
| `--scale` | 缩放系数 | 1.0 |
| `--no-compress` | 不压缩输出视频 | 压缩 |

## 性能

| 渲染方式 | FPS (80x45) | 说明 |
|---------|-------------|------|
| PyTorch GPU | ~13.6 | 需 NVIDIA 显卡 + CUDA |
| NumPy 向量化 | ~11.3 | CPU 自动降级 |
| CPU Fast (PIL) | ~9.5 | 兼容模式 |
| 原始纯 CPU | ~2.0 | 优化前 |

硬件: RTX 3060 Laptop 6GB, CUDA 12.1

## 技术架构

```
输入视频 ---> OpenCV 读取帧 ---> 帧队列 ---> 批处理渲染 ---> 写入队列 ---> ffmpeg 编码输出
                  |                          |
           +------+------+           +------+------+
           | 灰度/采样    |           | PyTorch GPU |
           | 字符映射     |           | NumPy CPU   |
           | 颜色渲染     |           | Numba (可选)|
           +-------------+           +-------------+
```

### 核心优化

- **三层渲染降级**: PyTorch CUDA -> NumPy 向量化 -> PIL CPU
- **帧批处理**: 滑动窗口调度, batch_size=16, max_workers=2
- **Atlas 缓存**: 预构建字符纹理图集，避免重复渲染
- **自动编码检测**: 依据 CUDA/字体可用性自动选择最优路径

## 项目结构

```
ascii-art-warp/
+-- ascii_art_warp_final.py   # 主程序
+-- install.bat               # Windows 一键安装脚本
+-- README.md                 # 本文件
+-- LICENSE                   # MIT 许可证
```

## 依赖

### 核心依赖

- Python 3.10+
- numpy
- opencv-python (cv2)
- pillow (PIL)
- pygame（GUI 和 --watch 模式）
- ffmpeg（视频编码）

### GPU 加速（可选）

- torch + torchvision (CUDA 版)
- NVIDIA 显卡 + CUDA 11.8+

## 致谢

本项目基于 [image2textart](https://github.com/WindowGenerator/image2textart)（作者：WindowGenerator）二次开发，在原项目基础上新增了 GPU 加速渲染、帧批处理调度、视频压缩、GUI/CLI/终端播放等完整功能。

感谢以下开源项目为本项目提供的技术支撑：

- **[FFmpeg](https://ffmpeg.org/)** — 视频编码与压缩引擎（LGPL/GPL 许可）
- **[PyTorch](https://pytorch.org/)** — GPU 加速计算框架（BSD 许可）
- **[OpenCV](https://opencv.org/)** — 视频帧读取与图像处理（Apache 2.0 许可）
- **[Pillow](https://python-pillow.org/)** — 字体渲染与图像处理（Historical Permission Notice）
- **[NumPy](https://numpy.org/)** — 向量化数值计算（BSD 许可）
- **[pygame](https://www.pygame.org/)** — GUI 界面框架（LGPL 许可）

## 常见问题

**Q: 导出视频颜色偏暗？**

A: 使用 `--exposure 1.6` 提高曝光度，或在 GUI 中调整「曝光度」滑块。

**Q: GPU 加速不生效？**

A: 确保安装了 CUDA 版 PyTorch。`install.bat` 会自动检测并选择合适版本。

**Q: 终端播放（--watch）颜色不对？**

A: 确保终端支持 ANSI 24-bit 真彩色（Windows Terminal / VSCode 终端 / iTerm2 均支持）。

## License

MIT License — 基于 [image2textart](https://github.com/WindowGenerator/image2textart)