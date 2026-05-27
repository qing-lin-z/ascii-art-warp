# ASCII Art Warp

GPU-accelerated video to ASCII art converter with PyTorch CUDA support.

基于 [image2textart](https://github.com/WindowGenerator/image2textart) 二次开发，支持 GPU 加速和视频压缩。

## 功能特性

- **GPU 加速**: PyTorch CUDA 像素映射 + 渲染，比纯 CPU 快 6-7 倍
- **多 GPU 支持**: 自动检测，支持 NVIDIA GPU 选择
- **视频压缩**: 内置 ffmpeg 压缩，文件体积减少 70%+
- **多进程处理**: 滑动窗口批处理，充分利用多核 CPU
- **彩色/灰度**: 支持彩色 ASCII 艺术和灰度模式
- **多种加速后端**: PyTorch CUDA → Numba Parallel → NumPy Vectorized 三层降级

## 安装依赖

```bash
pip install opencv-python pillow numpy numba torch
```

或使用 conda:
```bash
conda install opencv pillow numpy numba pytorch torchvision -c pytorch -c conda-forge
```

## 使用方法

```python
import ascii_art_warp_final as aaf

# 视频转换
aaf.video_to_mkv_multiprocess(
    input_path='input.mp4',
    output_path='output.mkv',
    target_w=80,           # 输出宽度（字符数）
    target_h=45,           # 输出高度（字符数）
    charset=' .:-=+*#%@',  # 字符集
    font_path='consola.ttf',
    font_size=16,
    use_color=True,        # 彩色模式
    max_workers=4,
    compress=True,         # 启用 ffmpeg 压缩
    crf=23,               # 压缩质量 (0-51)
    accel_mode='torch_cuda_single',  # GPU 加速模式
    gpu_id=0
)

# 图片转换
aaf.image_to_image(
    input_path='input.jpg',
    output_path='output.png',
    target_w=80,
    target_h=45,
    charset=' .:-=+*#%@',
    font_path='consola.ttf',
    font_size=16,
    bg_color_rgb=(0, 0, 0),
    use_color=True
)
```

## 性能基准

| 配置 | 分辨率 | FPS |
|------|--------|-----|
| CPU Fast | 80x45 | ~9.5 |
| NumPy Vectorized | 80x45 | ~11.3 |
| **PyTorch CUDA** | 80x45 | **~13.6** |

测试环境: RTX 3060 Laptop 6GB, Intel i7-10750H, 120帧 640x360 彩色视频

## 加速模式

- `torch_cuda_single`: PyTorch CUDA 单卡（推荐）
- `torch_cuda_dataparallel`: PyTorch CUDA 多卡
- `torch_cpu`: PyTorch CPU
- `numba_parallel`: Numba 并行
- `numpy_vectorized`: NumPy 向量化

## 项目结构

```
ascii_art_warp_final.py  # 主程序
ascii_art_warp_final.spec # PyInstaller 配置
README.md
```

## License

MIT License - 基于 [image2textart](https://github.com/WindowGenerator/image2textart)
