import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import threading
import os
import time
import subprocess
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from queue import Queue

# ========================== GPU/加速后端检测（静默，只设变量）==========================
_ACCEL_OPTIONS = []
_ACCEL_CURRENT = "numpy_vectorized"
_HAS_TORCH_CUDA = False
_TORCH_DEVICES = []

# ---- PyTorch CUDA 检测 ----
try:
    import torch as _torch
    _HAS_TORCH_CUDA = _torch.cuda.is_available()
    if _HAS_TORCH_CUDA:
        for i in range(_torch.cuda.device_count()):
            props = _torch.cuda.get_device_properties(i)
            name = getattr(props, 'name', f'GPU {i}')
            vram = props.total_memory / (1024**3)
            compute = f'{props.major}.{props.minor}'
            _TORCH_DEVICES.append((i, name, round(vram, 1), compute))
except:
    pass

# ---- CuPy 检测 ----
_HAS_CUPY = False
try:
    import cupy
    _HAS_CUPY = True
except:
    pass

# ---- Numba 检测 ----
_HAS_NUMBA = False
_NUMBA_VERSION = ""
try:
    import numba as _nb
    _HAS_NUMBA = True
    _NUMBA_VERSION = _nb.__version__
except:
    pass

# ---- 构建可选后端列表 ----
_ACCEL_OPTIONS = []
if _HAS_TORCH_CUDA:
    _ACCEL_OPTIONS.append(("torch_cuda_single", "PyTorch CUDA (单GPU)", "使用选定显卡"))
    if len(_TORCH_DEVICES) > 1:
        _ACCEL_OPTIONS.append(("torch_dataparallel", "PyTorch DataParallel (多卡负载均衡)", f"自动分配到 {len(_TORCH_DEVICES)} 张卡"))
    _ACCEL_OPTIONS.append(("torch_cpu", "PyTorch CPU (多核)", "PyTorch 张量运算，单机多核"))
if _HAS_NUMBA:
    _ACCEL_OPTIONS.append(("numba_parallel", "Numba 并行 (CPU)", "JIT 并行化，自动 SIMD"))
_ACCEL_OPTIONS.append(("numpy_vectorized", "NumPy 向量化 (单线程)", "零依赖 fallback"))
if not _ACCEL_OPTIONS:
    _ACCEL_OPTIONS.append(("python_loop", "纯 Python (最慢)", "无任何加速库"))
_ACCEL_CURRENT = _ACCEL_OPTIONS[0][0] if _ACCEL_OPTIONS else "python_loop"

# ---- 仅在主进程打印一次 ----
# spawn 模式子进程的 __name__ 是 __mp_main__，被 import 时是模块名，都排除
_IN_MAIN = __name__ == '__main__'
if _IN_MAIN:
    _msg = []
    if _HAS_TORCH_CUDA:
        for d in _TORCH_DEVICES:
            _msg.append(f"  [{d[0]}] {d[1]}  {d[2]} GB  Compute {d[3]}")
        print(f"[GPU] PyTorch CUDA 可用: {len(_TORCH_DEVICES)} 台设备")
        print("\n".join(_msg))
    if _HAS_NUMBA:
        print(f"[GPU] Numba 可用: {_NUMBA_VERSION}")
    if not _HAS_TORCH_CUDA and not _HAS_NUMBA:
        print("[GPU] 无 GPU/Numba，使用 NumPy 向量化")
    print(f"[GPU] 默认后端: {_ACCEL_OPTIONS[0][1] if _ACCEL_OPTIONS else '纯 Python'}")

# ========================== 多进程启动方式 ==========================
if sys.platform == 'win32':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# ========================== 检测 ffmpeg ==========================
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

# ========================== 字体检测工具 ==========================
def get_system_monospace_font():
    """检测系统默认等宽字体"""
    candidates = []
    if os.name == 'nt':
        for name in ["Consolas", "Courier New", "Lucida Console", "Monaco"]:
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
                try:
                    i = 0
                    while True:
                        val, data = winreg.EnumValue(key, i)
                        if name.lower() in val.lower() and data.lower().endswith(('.ttf', '.ttc')):
                            path = os.path.join(os.environ['WINDIR'], 'Fonts', os.path.basename(data))
                            if os.path.exists(path):
                                candidates.append(path)
                        i += 1
                except:
                    pass
                winreg.CloseKey(key)
            except:
                pass
    if sys.platform == 'darwin':
        candidates.extend(["/System/Library/Fonts/Monaco.ttf",
                          "/System/Library/Fonts/Courier.ttc"])
    if sys.platform.startswith('linux'):
        candidates.extend(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                          "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"])
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

# ========================== 像素→字符映射（多后端）==========================
# 全局路由函数
_map_pixels_to_chars = None
_map_pixels_to_colored = None
_ACCEL_MODE = "NumPy 向量化 (单线程)"

def _map_torch_cuda(gray_array, charset, scale):
    import torch
    t = torch.from_numpy(gray_array).float().cuda()
    idx = torch.clamp((t / scale).long(), 0, len(charset) - 1)
    chars = np.frombuffer(charset.encode('utf-8'), dtype='S1').reshape(-1)
    return chars[idx.cpu().numpy()]

def _map_torch_cuda_batch(gray_arrays, charset, scale):
    """gray_arrays: list of (H,W) uint8 numpy arrays → list of char matrices"""
    import torch
    try:
        B = len(gray_arrays)
        h, w = gray_arrays[0].shape
        # Stack into (B, H, W) CUDA tensor in one transfer
        stacked = torch.from_numpy(np.stack(gray_arrays, axis=0)).float().cuda()  # (B,H,W)
        idx = torch.clamp((stacked / scale).long(), 0, len(charset) - 1)        # (B,H,W)
        # Safe charset: use ord() for Unicode chars, avoid utf-8 byte truncation
        chars = np.array([c for c in charset], dtype='U1')
        results = []
        for b in range(B):
            results.append(chars[idx[b].cpu().numpy()])
        del stacked, idx
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"_map_torch_cuda_batch failed: {e}") from e

def _map_torch_cuda_colored(gray_array, color_array, charset, scale):
    return _map_torch_cuda(gray_array, charset, scale), color_array

def _map_torch_cuda_colored_batch(gray_arrays, color_arrays, charset, scale):
    """gray_arrays, color_arrays: list of (H,W) / (H,W,3) arrays → list of (char_matrices, color_maps)"""
    chars_results = _map_torch_cuda_batch(gray_arrays, charset, scale)
    return [(chars_results[i], color_arrays[i]) for i in range(len(chars_results))]

def _map_torch_cpu_colored(gray_array, color_array, charset, scale):
    return _map_torch_cpu(gray_array, charset, scale), color_array

def _map_torch_cpu(gray_array, charset, scale):
    import torch
    t = torch.from_numpy(gray_array).float()
    idx = torch.clamp((t / scale).long(), 0, len(charset) - 1)
    chars = np.frombuffer(charset.encode('utf-8'), dtype='S1').reshape(-1)
    return chars[idx.numpy()]

def _map_numba_parallel(gray_array, charset, scale):
    from numba import njit, prange
    h, w = gray_array.shape
    n = len(charset)
    out = np.empty((h, w), dtype=np.int32)
    @njit(parallel=True, fastmath=True, cache=True)
    def fill(gray, result):
        for y in prange(h):
            for x in range(w):
                v = int(gray[y, x] / scale)
                result[y, x] = n - 1 if v >= n else (0 if v < 0 else v)
    fill(gray_array, out)
    chars = np.frombuffer(charset.encode('utf-8'), dtype='S1').reshape(-1)
    return chars[out]

def _map_numba_parallel_colored(gray_array, color_array, charset, scale):
    return _map_numba_parallel(gray_array, charset, scale), color_array

def _map_numpy(gray_array, charset, scale):
    chars = np.frombuffer(charset.encode('utf-8'), dtype='S1').reshape(-1)
    n = len(chars)
    idx = np.clip((gray_array.astype(np.float32) / scale).astype(np.int32), 0, n - 1)
    return chars[idx]

def _map_numpy_colored(gray_array, color_array, charset, scale):
    return _map_numpy(gray_array, charset, scale), color_array

_ROUTES = {
    "torch_cuda_single":   (_map_torch_cuda,         _map_torch_cuda_colored,        "PyTorch CUDA (单GPU)"),
    "torch_dataparallel":   (_map_torch_cpu,            _map_torch_cpu_colored,         "PyTorch DataParallel (多卡)"),
    "torch_cpu":            (_map_torch_cpu,            _map_torch_cpu_colored,         "PyTorch CPU (多核)"),
    "numba_parallel":       (_map_numba_parallel,      _map_numba_parallel_colored,    "Numba 并行 (CPU)"),
    "numpy_vectorized":    (_map_numpy,                _map_numpy_colored,             "NumPy 向量化"),
    "python_loop":         (_map_numpy,                _map_numpy_colored,             "纯 Python"),
}

_map_pixels_to_chars_batch = None   # GPU batch pixel mapper (grayscale)
_map_pixels_to_colored_batch = None  # GPU batch pixel mapper (colored)

def set_accel_mode(mode_key):
    global _map_pixels_to_chars, _map_pixels_to_colored, _ACCEL_MODE
    global _map_pixels_to_chars_batch, _map_pixels_to_colored_batch
    if mode_key in _ROUTES:
        gfn, cfn, label = _ROUTES[mode_key]
        _map_pixels_to_chars = gfn
        _map_pixels_to_colored = cfn
        _ACCEL_MODE = label
        # Batch GPU mappers
        if mode_key == "torch_cuda_single":
            _map_pixels_to_chars_batch = _map_torch_cuda_batch
            _map_pixels_to_colored_batch = _map_torch_cuda_colored_batch
        else:
            _map_pixels_to_chars_batch = None
            _map_pixels_to_colored_batch = None

# 初始化默认后端（函数定义之后调用）
set_accel_mode(_ACCEL_CURRENT)

# ========================== 核心转换函数 ==========================
def image_to_colored_matrix_fast(pil_color, target_w, target_h, charset):
    if len(charset) < 2:
        raise ValueError("字符集至少需要 2 个字符")
    img = pil_color.resize((target_w, target_h), Image.Resampling.LANCZOS)
    gray_data = np.array(img.convert("L"), dtype=np.uint8)
    color_data = np.array(img, dtype=np.uint8)
    scale = 255.0 / max(1, len(charset) - 1)
    mat, cmap = _map_pixels_to_colored(gray_data, color_data, charset, scale)
    return mat, cmap, target_h, target_w

def image_to_grayscale_matrix_fast(gray_pil, target_w, target_h, charset):
    if len(charset) < 2:
        raise ValueError("字符集至少需要 2 个字符")
    gray_data = np.array(gray_pil, dtype=np.uint8)
    scale = 255.0 / max(1, len(charset) - 1)
    mat = _map_pixels_to_chars(gray_data, charset, scale)
    return mat, target_h, target_w

# ========================== Sprite 渲染引擎 ==========================
_SPRITE_CACHE = {}

def _render_sprites_gray(font, char_width, char_height, charset_bytes):
    sprites = {}
    for ch in charset_bytes:
        img = Image.new("RGB", (char_width, char_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((0, 0), ch.decode('utf-8'), font=font, fill=(255, 255, 255))
        except:
            draw.text((0, 0), '?', font=font, fill=(255, 255, 255))
        sprites[ch] = img
    return sprites

def _render_sprites_color(font, char_width, char_height, charset_bytes):
    sprites = {}
    for ch in charset_bytes:
        img = Image.new("RGBA", (char_width, char_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((0, 0), ch.decode('utf-8'), font=font, fill=(255, 255, 255, 255))
        except:
            draw.text((0, 0), '?', font=font, fill=(255, 255, 255, 255))
        sprites[ch] = img
    return sprites

def _ensure_sprites(font, char_width, char_height, charset_bytes, use_color):
    # charset_bytes must be tuple (not np.array) for dict key hashability
    if isinstance(charset_bytes, np.ndarray):
        charset_key = tuple(charset_bytes.tolist())
    else:
        charset_key = tuple(charset_bytes)
    key = (charset_key, char_width, char_height, use_color)
    if key not in _SPRITE_CACHE:
        if use_color:
            _SPRITE_CACHE[key] = _render_sprites_color(font, char_width, char_height, charset_bytes)
        else:
            _SPRITE_CACHE[key] = _render_sprites_gray(font, char_width, char_height, charset_bytes)
    return _SPRITE_CACHE[key]

def render_colored_ascii_sprite_fast(matrix, color_map, font_path, font_size, bg_color, char_dims=None):
    """NumPy 完全向量化彩色渲染，无 Python 循环"""
    font_path = os.path.abspath(font_path)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        fallback = get_system_monospace_font()
        if fallback and os.path.exists(fallback):
            try:
                font = ImageFont.truetype(fallback, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

    if char_dims:
        char_width, char_height = char_dims
    else:
        bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
        char_width = max(1, bbox[2] - bbox[0])
        char_height = max(1, bbox[3] - bbox[1])

    rows, cols = matrix.shape

    # 预渲染所有字符 sprite 为 numpy 数组
    unique_ch = tuple(set(matrix.flat))
    charset_bytes = np.array([c if isinstance(c, bytes) else bytes([ord(c)]) for c in unique_ch], dtype='S1')
    sprites_pil = _ensure_sprites(font, char_width, char_height, charset_bytes, True)

    # 构建 atlas: (num_chars, char_h, char_w, 4)
    char_list = list(sprites_pil.keys())
    atlas = np.stack([np.array(sprites_pil[ch], dtype=np.uint8) for ch in char_list])  # (N, char_h, char_w, 4)
    char_to_idx = {ch: i for i, ch in enumerate(char_list)}

    # 把 matrix 转成索引 (rows, cols)
    vec_encode = np.vectorize(lambda x: bytes([ord(x)]) if isinstance(x, str) else (bytes(x) if isinstance(x, np.bytes_) else x))
    mat_bytes = vec_encode(matrix)
    mat_indices = np.array([[char_to_idx.get(ch, 0) for ch in row] for row in mat_bytes], dtype=np.int64)  # (rows, cols)

    # 从 atlas 取 sprite: (rows, cols, char_h, char_w, 4)
    sprites = atlas[mat_indices]  # (rows, cols, char_h, char_w, 4)

    # 分离 alpha 和 rgb
    alpha = sprites[:, :, :, :, 3:4].astype(np.float32) / 255.0  # (rows, cols, char_h, char_w, 1)
    sprite_rgb = sprites[:, :, :, :, :3].astype(np.float32)  # (rows, cols, char_h, char_w, 3)

    # 颜色扩展到每个像素: (rows, cols, 1, 1, 3)
    color = color_map.astype(np.float32).reshape(rows, cols, 1, 1, 3)

    # 修复: 直接使用原图颜色作为字符颜色
    # 字符区域(alpha>0)完全显示原图颜色，只有完全透明区才显示背景色
    colored = np.broadcast_to(color, (rows, cols, char_height, char_width, 3)).copy()

    # 背景色
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 1, 1, 3)
    # 二值化 alpha: 有笔画的地方完全不混背景色，只有纯透明区域才用背景
    mask = (alpha > 0.5).astype(np.float32)
    blended = (mask * colored + (1.0 - mask) * bg).clip(0, 255).astype(np.uint8)

    # reshape 为图像: (rows*char_h, cols*char_w, 3)
    img = blended.transpose(0, 2, 1, 3, 4).reshape(rows * char_height, cols * char_width, 3)

    return Image.fromarray(img, mode="RGB")

def render_grayscale_ascii_sprite_fast(matrix, font_path, font_size, fg_color, bg_color, char_dims=None):
    """NumPy 完全向量化灰度渲染，无 Python 循环"""
    font_path = os.path.abspath(font_path)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        fallback = get_system_monospace_font()
        if fallback and os.path.exists(fallback):
            try:
                font = ImageFont.truetype(fallback, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

    if char_dims:
        char_width, char_height = char_dims
    else:
        bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
        char_width = max(1, bbox[2] - bbox[0])
        char_height = max(1, bbox[3] - bbox[1])

    rows, cols = matrix.shape

    unique_ch = tuple(set(matrix.flat))
    charset_bytes = np.array([c if isinstance(c, bytes) else bytes([ord(c)]) for c in unique_ch], dtype='S1')
    sprites_pil = _ensure_sprites(font, char_width, char_height, charset_bytes, False)

    # 构建 atlas: (num_chars, char_h, char_w, 3)
    char_list = list(sprites_pil.keys())
    atlas = np.stack([np.array(sprites_pil[ch], dtype=np.uint8) for ch in char_list])  # (N, char_h, char_w, 3)
    char_to_idx = {ch: i for i, ch in enumerate(char_list)}

    # 把 matrix 转成索引
    vec_encode = np.vectorize(lambda x: bytes([ord(x)]) if isinstance(x, str) else (bytes(x) if isinstance(x, np.bytes_) else x))
    mat_bytes = vec_encode(matrix)
    mat_indices = np.array([[char_to_idx.get(ch, 0) for ch in row] for row in mat_bytes], dtype=np.int64)  # (rows, cols)

    # 从 atlas 取 sprite: (rows, cols, char_h, char_w, 3)
    sprites = atlas[mat_indices]  # (rows, cols, char_h, char_w, 3)

    # 颜色混合
    fg = np.array(fg_color, dtype=np.float32).reshape(1, 1, 1, 1, 3)
    colored = (sprites.astype(np.float32) * fg / 255.0).clip(0, 255).astype(np.uint8)  # (rows, cols, char_h, char_w, 3)

    # reshape 为图像
    img = colored.transpose(0, 2, 1, 3, 4).reshape(rows * char_height, cols * char_width, 3)

    return Image.fromarray(img, mode="RGB")

def render_colored_ascii_sprite_torch(matrix, color_map, font_path, font_size, bg_color, char_dims=None):
    """PyTorch GPU 渲染，完全在 GPU 上组装 ASCII 图像"""
    import torch
    global _font_cache
    fc = _font_cache
    device = torch.device(fc.get('torch_device', 'cuda:0') if fc.get('has_cuda') else 'cpu')

    if char_dims:
        char_width, char_height = char_dims
    else:
        char_width, char_height = fc['char_dims']

    rows, cols = matrix.shape

    # 使用缓存的 atlas
    atlas = fc.get('torch_atlas')
    char_to_idx = fc.get('torch_char_to_idx')
    # 从 atlas 形状获取实际字符尺寸
    _, char_height, char_width, _ = atlas.shape
    if atlas is None or char_to_idx is None:
        # fallback: 使用 CPU fast 版本
        return render_colored_ascii_sprite_fast(matrix, color_map, font_path, font_size, bg_color, char_dims)

    # 把 matrix 转成索引（向量化）
    vec_encode = np.vectorize(lambda x: bytes([ord(x)]) if isinstance(x, str) else (bytes(x) if isinstance(x, np.bytes_) else x))
    mat_bytes = vec_encode(matrix)
    mat_indices = np.array([[char_to_idx.get(ch, 0) for ch in row] for row in mat_bytes], dtype=np.int64)
    mat_indices_t = torch.from_numpy(mat_indices).to(device)  # (rows, cols)
    color_map_t = torch.from_numpy(color_map.astype(np.float32)).to(device)  # (rows, cols, 3)

    # 从 atlas 取 sprite: (rows, cols, char_h, char_w, 4)
    sprites_selected = atlas[mat_indices_t]  # (rows, cols, char_h, char_w, 4)

    # 分离 alpha 和 rgb
    alpha = sprites_selected[:, :, :, :, 3:4].float() / 255.0  # (rows, cols, char_h, char_w, 1)
    sprite_rgb = sprites_selected[:, :, :, :, :3].float()  # (rows, cols, char_h, char_w, 3)

    # 颜色: (rows, cols, 1, 1, 3)
    color = color_map_t.unsqueeze(2).unsqueeze(3)  # (rows, cols, 1, 1, 3)

    # 修复: 直接使用原图颜色作为字符颜色
    # 字符区域(alpha>0)完全显示原图颜色，只有完全透明区才显示背景色
    colored = color.expand(sprites_selected.shape[0], sprites_selected.shape[1], char_height, char_width, 3)

    # 背景色
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device).view(1, 1, 1, 1, 3)
    # 二值化 alpha: 有笔画的地方完全不混背景色
    mask = (alpha > 0.5).float()
    blended = (mask * colored + (1.0 - mask) * bg).clamp(0, 255).byte()

    # reshape 为图像: (rows*char_h, cols*char_w, 3)
    img = blended.permute(0, 2, 1, 3, 4).contiguous()
    img = img.view(rows * char_height, cols * char_width, 3)

    # 转回 CPU 和 PIL
    img_np = img.cpu().numpy()
    return Image.fromarray(img_np, mode="RGB")

# 保留旧接口，但内部调用 fast 版本
render_colored_ascii_sprite = render_colored_ascii_sprite_fast
render_grayscale_ascii_sprite = render_grayscale_ascii_sprite_fast

# ========================== 进程池字体缓存 ==========================
_font_cache = {}

def _init_process_pool(font_path, font_size, target_w, target_h, charset, use_color, accel_mode, gpu_id, batch_size=4):
    global _font_cache
    font_path = os.path.abspath(font_path)

    _font_cache['font_path'] = font_path
    _font_cache['font_size'] = font_size
    _font_cache['target_w'] = target_w
    _font_cache['target_h'] = target_h
    _font_cache['charset'] = charset
    _font_cache['use_color'] = use_color
    _font_cache['gpu_id'] = gpu_id
    _font_cache['accel_mode'] = accel_mode
    _font_cache['batch_size'] = batch_size

    try:
        _font_cache['font'] = ImageFont.truetype(font_path, font_size)
    except:
        fallback = get_system_monospace_font()
        if fallback and os.path.exists(fallback):
            try:
                _font_cache['font'] = ImageFont.truetype(fallback, font_size)
            except:
                _font_cache['font'] = ImageFont.load_default()
        else:
            _font_cache['font'] = ImageFont.load_default()

    temp_img = Image.new("RGB", (1,1))
    temp_draw = ImageDraw.Draw(temp_img)
    try:
        total_width = sum(_font_cache['font'].getlength(c) for c in "MWAy0g@#")
        char_width = max(1, int(total_width / 8))
    except:
        bbox = temp_draw.textbbox((0,0), "A", font=_font_cache['font'])
        char_width = max(1, bbox[2] - bbox[0])
    bbox = temp_draw.textbbox((0,0), "A", font=_font_cache['font'])
    char_height = max(1, bbox[3] - bbox[1])
    _font_cache['char_dims'] = (char_width, char_height)

    charset_bytes = tuple(set(c.encode('utf-8') if isinstance(c, str) else c for c in charset))
    _font_cache['sprites'] = _ensure_sprites(_font_cache['font'], char_width, char_height, charset_bytes, use_color)

    # 设置 CUDA device
    _font_cache['has_cuda'] = False
    if accel_mode in ("torch_cuda_single", "torch_dataparallel") and _HAS_TORCH_CUDA:
        try:
            import torch
            torch.cuda.set_device(gpu_id)
            _font_cache['has_cuda'] = True
            _font_cache['torch_device'] = f"cuda:{gpu_id}"
        except Exception:
            _font_cache['has_cuda'] = False

    # 预构建 torch atlas 用于 GPU 渲染
    _font_cache['torch_atlas'] = None
    _font_cache['torch_char_to_idx'] = None
    if _font_cache['has_cuda']:
        try:
            import torch
            device = torch.device(_font_cache['torch_device'])
            sprites_pil = _font_cache['sprites']
            char_list = list(sprites_pil.keys())
            atlas = torch.stack([torch.from_numpy(np.array(sprites_pil[ch], dtype=np.uint8)) for ch in char_list]).to(device)
            _font_cache['torch_atlas'] = atlas
            _font_cache['torch_char_to_idx'] = {ch: i for i, ch in enumerate(char_list)}
        except Exception:
            pass

    set_accel_mode(accel_mode)

def process_frame_from_cache(frame_bgr):
    global _font_cache
    fc = _font_cache
    frame_resized = cv2.resize(frame_bgr, (fc['target_w'], fc['target_h']))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    if fc['use_color']:
        matrix, color_map, _, _ = image_to_colored_matrix_fast(pil_img, fc['target_w'], fc['target_h'], fc['charset'])
        # 优先使用 torch GPU 渲染
        if fc.get('has_cuda') and fc.get('torch_atlas') is not None:
            ascii_img = render_colored_ascii_sprite_torch(matrix, color_map, fc['font_path'], fc['font_size'],
                                                           (0,0,0), fc['char_dims'])
        else:
            ascii_img = render_colored_ascii_sprite(matrix, color_map, fc['font_path'], fc['font_size'],
                                                     (0,0,0), fc['char_dims'])
    else:
        gray_pil = pil_img.convert("L")
        matrix, _, _ = image_to_grayscale_matrix_fast(gray_pil, fc['target_w'], fc['target_h'], fc['charset'])
        ascii_img = render_grayscale_ascii_sprite(matrix, fc['font_path'], fc['font_size'],
                                                   (255,255,255), (0,0,0), fc['char_dims'])

    return cv2.cvtColor(np.array(ascii_img), cv2.COLOR_RGB2BGR)

def process_frame_batch(frames):
    """frames: list of BGR numpy arrays → list of BGR processed images
    GPU batch: resize+convert all frames, batch GPU pixel mapping, then render each."""
    global _font_cache, _map_pixels_to_colored_batch
    fc = _font_cache
    tw, th = fc['target_w'], fc['target_h']
    charset = fc['charset']
    scale = 255.0 / max(1, len(charset) - 1)

    # ── Step 1: resize + color-convert all frames ──────────────────────
    gray_list, color_list = [], []
    pil_list = []
    for f in frames:
        if f.shape[1] != tw or f.shape[0] != th:
            f_r = cv2.resize(f, (tw, th))
        else:
            f_r = f
        f_rgb = cv2.cvtColor(f_r, cv2.COLOR_BGR2RGB)
        pil_list.append(Image.fromarray(f_rgb))
        gray_list.append(np.array(pil_list[-1].convert("L"), dtype=np.uint8))
        color_list.append(np.array(pil_list[-1], dtype=np.uint8))

    # ── Step 2: batched GPU pixel mapping ───────────────────────────────
    if fc['use_color']:
        try:
            if '_map_pixels_to_colored_batch' in globals() and _map_pixels_to_colored_batch is not None:
                results = _map_pixels_to_colored_batch(gray_list, color_list, charset, scale)
            else:
                raise RuntimeError("_map_pixels_to_colored_batch not available, using fallback")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] batch colored failed, fallback to single-frame: {e}")
            results = []
            for g, c in zip(gray_list, color_list):
                m, cmap = _map_pixels_to_colored(g, c, charset, scale)
                results.append((m, cmap))
    else:
        try:
            if '_map_pixels_to_chars_batch' in globals() and _map_pixels_to_chars_batch is not None:
                matrices = _map_pixels_to_chars_batch(gray_list, charset, scale)
                results = [(m, None) for m in matrices]
            else:
                raise RuntimeError("_map_pixels_to_chars_batch not available, using fallback")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] batch grayscale failed, fallback to single-frame: {e}")
            results = []
            for g in gray_list:
                m = _map_pixels_to_chars(g, charset, scale)
                results.append((m, None))

    # ── Step 3: render each frame ───────────────────────────────────────
    outputs = []
    use_torch = fc.get('has_cuda') and fc.get('torch_atlas') is not None
    for mat, cmap in results:
        if fc['use_color']:
            if use_torch:
                img = render_colored_ascii_sprite_torch(mat, cmap, fc['font_path'], fc['font_size'],
                                                          (0,0,0), fc['char_dims'])
            else:
                img = render_colored_ascii_sprite(mat, cmap, fc['font_path'], fc['font_size'],
                                                   (0,0,0), fc['char_dims'])
        else:
            img = render_grayscale_ascii_sprite(mat, fc['font_path'], fc['font_size'],
                                                 (255,255,255), (0,0,0), fc['char_dims'])
        outputs.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return outputs

def process_frame(args):
    """纯 CPU 处理（兼容旧接口）"""
    frame_bgr, target_w, target_h, charset, font_path, font_size, bg_color_rgb, use_color = args
    font_path = os.path.abspath(font_path)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        fallback = get_system_monospace_font()
        if fallback and os.path.exists(fallback):
            try:
                font = ImageFont.truetype(fallback, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

    try:
        total_width = sum(font.getlength(c) for c in "MWAy0g@#")
        char_width = max(1, int(total_width / 8))
    except:
        bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
        char_width = max(1, bbox[2] - bbox[0])
    bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
    char_height = max(1, bbox[3] - bbox[1])
    char_dims = (char_width, char_height)

    # 如果生产者已缩小帧（reduce_video=True），跳过重复 resize
    if frame_bgr.shape[1] != target_w or frame_bgr.shape[0] != target_h:
        frame_resized = cv2.resize(frame_bgr, (target_w, target_h))
    else:
        frame_resized = frame_bgr
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    if use_color:
        matrix, color_map, _, _ = image_to_colored_matrix_fast(pil_img, target_w, target_h, charset)
        ascii_img = render_colored_ascii_sprite(matrix, color_map, font_path, font_size, bg_color_rgb, char_dims)
    else:
        gray_pil = pil_img.convert("L")
        matrix, _, _ = image_to_grayscale_matrix_fast(gray_pil, target_w, target_h, charset)
        ascii_img = render_grayscale_ascii_sprite(matrix, font_path, font_size, (255,255,255), bg_color_rgb, char_dims)

    return cv2.cvtColor(np.array(ascii_img), cv2.COLOR_RGB2BGR)

# ========================== 生产者函数 ==========================
def producer_func(input_path, frame_queue, done_event, buffer_size, reduce_video=False, target_dim=None):
    """
    生产者：读取视频帧放入队列。
    reduce_video=True 时，在放入队列前缩小帧到 target_dim（目标 ASCII 分辨率），
    大幅减少进程间传输的数据量。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        done_event.set()
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if reduce_video and target_dim:
            frame = cv2.resize(frame, target_dim, interpolation=cv2.INTER_AREA)
        frame_queue.put((idx, frame))
        idx += 1
    cap.release()
    done_event.set()

# ========================== 异步写盘线程 ==========================
_writer_queue = None
_writer_active = False
_writer_thread = None
_writer_out = None   # cv2.VideoWriter 实例在同一进程，直接引用不用队列传


def _compress_video_ffmpeg(input_path, output_path, crf=23, preset="medium", progress_callback=None):
    """使用 ffmpeg 压缩视频
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（压缩后）
        crf: 质量参数，0-51，越小质量越高（默认23）
        preset: 编码速度预设 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        progress_callback: 可选，进度回调 (current_sec, total_sec)
    """
    import subprocess

    # 获取视频时长
    total_sec = 0
    try:
        probe = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', input_path
        ], capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            import json
            info = json.loads(probe.stdout)
            total_sec = float(info.get('format', {}).get('duration', 0))
    except:
        pass

    cmd = [
        'ffmpeg', '-y',
        '-progress', 'pipe:1',   # 结构化进度输出到 stdout
        '-nostats',               # 禁止 stderr 的默认进度行
        '-loglevel', 'error',     # stderr 只输出错误
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True
        )
        # 解析 -progress pipe:1 的 key=value 输出
        # 优先 out_time_us，其次 out_time_ms，最后解析 out_time=HH:MM:SS.xxx
        for line in process.stdout:
            line = line.strip()
            if line.startswith('out_time_us='):
                us = int(line.split('=', 1)[1])
                if us > 0 and progress_callback:
                    progress_callback(us / 1_000_000, total_sec if total_sec > 0 else 9999)
            elif line.startswith('out_time_ms='):
                ms_val = int(line.split('=', 1)[1])
                if ms_val > 0 and progress_callback:
                    progress_callback(ms_val / 1000, total_sec if total_sec > 0 else 9999)
            elif line.startswith('out_time='):
                try:
                    ts = line.split('=', 1)[1]
                    parts = ts.split(':')
                    sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    if sec > 0 and progress_callback:
                        progress_callback(sec, total_sec if total_sec > 0 else 9999)
                except:
                    pass

        process.wait()
        if process.returncode != 0:
            print(f"[WARN] ffmpeg 压缩失败: rc={process.returncode}")
            return False
        return True
    except FileNotFoundError:
        print("[WARN] ffmpeg 未找到，跳过压缩")
        return False
    except subprocess.TimeoutExpired:
        print("[WARN] ffmpeg 压缩超时")
        return False


def _writer_loop():
    global _writer_queue, _writer_active, _writer_out
    while _writer_active:
        try:
            item = _writer_queue.get(timeout=0.05)
            if item is None:
                break
            frame = item   # 不再从队列接收 out，直接用全局 _writer_out
            _writer_out.write(frame)
            _writer_queue.task_done()
        except:
            continue

# ========================== 多进程流式视频转换 ==========================
def video_to_mkv_multiprocess(input_path, output_path, target_w, target_h, charset, font_path, font_size,
                              bg_color_rgb, use_color, max_workers, buffer_size=30,
                              progress_callback=None, accel_mode="torch_cuda_single", gpu_id=0,
                              compress=False, crf=23, preset="medium",
                              reduce_video=True,
                              compress_progress_callback=None):
    global _writer_queue, _writer_active, _writer_thread
    _writer_queue = Queue(maxsize=buffer_size)
    _writer_active = True
    _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    _writer_thread.start()

    font_path = os.path.abspath(font_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        raise Exception("无法读取第一帧")

    try:
        test_frame = process_frame((first_frame, target_w, target_h, charset, font_path, font_size,
                                    bg_color_rgb, use_color))
    except Exception as e:
        raise Exception(f"测试帧处理失败: {str(e)}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_h, out_w = test_frame.shape[:2]

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mkv':
        for codec in ['X264', 'VP90', 'mp4v', 'avc1']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if out.isOpened():
                break
    elif ext == '.mp4':
        for codec in ['mp4v', 'avc1', 'X264']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if out.isOpened():
                break
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        raise Exception("无法初始化视频编码器")

    # out 必须对 writer 线程可见，且必须先于线程启动前设置
    global _writer_out
    _writer_out = out

    frame_queue = mp.Queue(maxsize=buffer_size)
    done_event = mp.Event()

    producer_args = (input_path, frame_queue, done_event, buffer_size,
                     reduce_video, (target_w, target_h) if reduce_video else None)
    producer_process = mp.Process(target=producer_func, args=producer_args, daemon=True)
    producer_process.start()

    pending = {}   # 乱序帧缓冲，key=idx，value=已处理但未按序显示的帧img
    next_idx = 0
    processed = 0
    start_time = time.time()
    max_pending = buffer_size * 3   # 仅用于限制内层 frame_queue 填充速度，不参与退出判断

    try:
        num_gpus = len(_TORCH_DEVICES) if (_HAS_TORCH_CUDA and accel_mode != "torch_cpu") else 1
        if accel_mode == "torch_dataparallel" and num_gpus > 1:
            print(f"[GPU] 多卡模式: {num_gpus} GPU, {max_workers} worker")

        init_args = (font_path, font_size, target_w, target_h, charset, use_color, accel_mode, gpu_id)
        batch_size = 16  # 增大 batch 以充分利用 GPU
        init_args_batch = init_args + (batch_size,)
        print(f"[INFO] batch={batch_size}, max_workers={max_workers}, accel={accel_mode}")
        # 滑动窗口：始终保持 max_workers * 4 个 batch 在飞，GPU 从不等待
        max_inflight = max_workers * 4
        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=_init_process_pool,
                                 initargs=init_args_batch) as executor:
            futures_map = {}   # future → (batch_idx, batch_frames)
            batch_start_idx = 0

            # 主循环：滑动窗口，GPU 不等待
            while futures_map or not done_event.is_set() or not frame_queue.empty():
                # 找出已完成的 batch
                done_keys = [f for f in list(futures_map.keys()) if f.done()]
                for f in done_keys:
                    batch_idx, batch_frames = futures_map.pop(f)
                    try:
                        results = f.result()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        results = [None] * len(batch_frames)
                    for ri, img in enumerate(results):
                        pending[batch_idx + ri] = img
                    while next_idx in pending:
                        img = pending.pop(next_idx)
                        if img is not None:
                            _writer_queue.put(img)
                        next_idx += 1
                        processed += 1
                        if progress_callback:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / processed if processed > 0 else 0
                            eta_seconds = avg_time * (total_frames - processed) if total_frames > processed else 0
                            eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0, eta_seconds)))
                            progress_callback(processed, total_frames, eta_str)

                # 补填滑动窗口
                while (len(futures_map) < max_inflight):
                    batch_frames = []
                    while len(batch_frames) < batch_size and not frame_queue.empty():
                        try:
                            _, frame = frame_queue.get_nowait()
                            batch_frames.append(frame)
                        except:
                            break
                    if not batch_frames:
                        break
                    future = executor.submit(process_frame_batch, batch_frames)
                    futures_map[future] = (batch_start_idx, batch_frames)
                    batch_start_idx += len(batch_frames)

                if not done_keys:
                    time.sleep(0.01)

                # 全部帧处理完毕且所有 batch 完成时退出
                if done_event.is_set() and frame_queue.empty() and not futures_map:
                    break
    finally:
        _writer_active = False
        _writer_queue.put(None)
        _writer_thread.join(timeout=10)
        out.release()
        producer_process.join(timeout=5)
        if producer_process.is_alive():
            producer_process.terminate()
        del first_frame, test_frame

    # ffmpeg 压缩（如启用）
    if compress:
        orig_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        print(f"[INFO] 开始 ffmpeg 压缩 (CRF={crf}, preset={preset})...")
        tmp_output = output_path + '.tmp_compress.mp4'
        if _compress_video_ffmpeg(output_path, tmp_output, crf=crf, preset=preset,
                                  progress_callback=compress_progress_callback):
            try:
                os.replace(tmp_output, output_path)
                new_size = os.path.getsize(output_path)
                ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
                print(f"[INFO] 压缩完成: {orig_size/1024/1024:.1f}MB → {new_size/1024/1024:.1f}MB (减少 {ratio:.1f}%)")
            except Exception as e:
                print(f"[WARN] 替换压缩文件失败: {e}")
        else:
            # 压缩失败，删除临时文件
            if os.path.exists(tmp_output):
                os.remove(tmp_output)

    return total_frames

# ========================== 图片转换 ==========================
def image_to_image(input_path, output_path, target_w, target_h, charset, font_path, font_size, bg_color_rgb, use_color):
    font_path = os.path.abspath(font_path)
    if len(charset) < 2:
        raise ValueError("字符集至少需要 2 个字符")

    pil_img = Image.open(input_path).convert("RGB")

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        fallback = get_system_monospace_font()
        if fallback and os.path.exists(fallback):
            try:
                font = ImageFont.truetype(fallback, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

    try:
        total_width = sum(font.getlength(c) for c in "MWAy0g@#")
        char_width = max(1, int(total_width / 8))
    except:
        bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
        char_width = max(1, bbox[2] - bbox[0])
    bbox = ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0), "A", font=font)
    char_height = max(1, bbox[3] - bbox[1])
    char_dims = (char_width, char_height)

    if use_color:
        matrix, color_map, _, _ = image_to_colored_matrix_fast(pil_img, target_w, target_h, charset)
        ascii_img = render_colored_ascii_sprite(matrix, color_map, font_path, font_size, bg_color_rgb, char_dims)
    else:
        gray_pil = pil_img.convert("L")
        matrix, _, _ = image_to_grayscale_matrix_fast(gray_pil, target_w, target_h, charset)
        ascii_img = render_grayscale_ascii_sprite(matrix, font_path, font_size, (255,255,255), bg_color_rgb, char_dims)

    ascii_img.save(output_path, "PNG")
    return True

# ========================== 音频合并 ==========================
def merge_audio(video_no_audio_path, original_video_path, output_with_audio_path):
    if not check_ffmpeg():
        return False, "ffmpeg 未安装"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-select_streams", "a", original_video_path],
                           capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        return False, "原视频没有音轨"
    cmd = ["ffmpeg", "-y", "-i", video_no_audio_path, "-i", original_video_path,
           "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_with_audio_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"ffmpeg 执行失败: {r.stderr[:300]}"
        return True, "成功"
    except Exception as e:
        return False, str(e)

# ========================== 预览悬浮窗 ==========================
class PreviewWindow:
    def __init__(self, parent, file_path, target_w, target_h, charset, font_path, font_size, bg_color_rgb, use_color):
        self.win = Toplevel(parent)
        self.win.title("字符画预览")
        self.win.geometry("800x600")
        self.file_path = file_path
        self.target_w = target_w
        self.target_h = target_h
        self.charset = charset
        self.font_path = os.path.abspath(font_path)
        self.font_size = font_size
        self.bg_color_rgb = bg_color_rgb
        self.use_color = use_color
        self.cancelled = False
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.canvas = tk.Canvas(self.win, bg='gray')
        sy = tk.Scrollbar(self.win, orient="vertical", command=self.canvas.yview)
        sx = tk.Scrollbar(self.win, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self.win.grid_rowconfigure(0, weight=1)
        self.win.grid_columnconfigure(0, weight=1)
        self.img_on_canvas = None
        self.load_preview()

    def on_close(self):
        self.cancelled = True
        self.win.destroy()

    def load_preview(self):
        def generate():
            if self.cancelled:
                return
            try:
                ext = os.path.splitext(self.file_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    pil_img = Image.open(self.file_path).convert("RGB")
                    if self.use_color:
                        matrix, color_map, _, _ = image_to_colored_matrix_fast(pil_img, self.target_w, self.target_h, self.charset)
                        ascii_img = render_colored_ascii_sprite(matrix, color_map, self.font_path, self.font_size, self.bg_color_rgb)
                    else:
                        gray_pil = pil_img.convert("L")
                        matrix, _, _ = image_to_grayscale_matrix_fast(gray_pil, self.target_w, self.target_h, self.charset)
                        ascii_img = render_grayscale_ascii_sprite(matrix, self.font_path, self.font_size, (255,255,255), self.bg_color_rgb)
                else:
                    cap = cv2.VideoCapture(self.file_path)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        raise Exception("无法读取视频帧")
                    pil_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if self.use_color:
                        matrix, color_map, _, _ = image_to_colored_matrix_fast(pil_rgb, self.target_w, self.target_h, self.charset)
                        ascii_img = render_colored_ascii_sprite(matrix, color_map, self.font_path, self.font_size, self.bg_color_rgb)
                    else:
                        gray_pil = pil_rgb.convert("L")
                        matrix, _, _ = image_to_grayscale_matrix_fast(gray_pil, self.target_w, self.target_h, self.charset)
                        ascii_img = render_grayscale_ascii_sprite(matrix, self.font_path, self.font_size, (255,255,255), self.bg_color_rgb)

                if self.cancelled:
                    return
                img_tk = ImageTk.PhotoImage(ascii_img)
                self.win.after(0, lambda: self.display_image(img_tk, ascii_img.size))
            except Exception as e:
                if not self.cancelled:
                    self.win.after(0, lambda: messagebox.showerror("预览错误", str(e)))

        threading.Thread(target=generate, daemon=True).start()

    def display_image(self, img_tk, original_size):
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, original_size[0], original_size[1]))
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

def watch_video(input_path, target_w, target_h, charset, font_path, font_size,
              use_color, accel_mode, gpu_id, reduce_video=True):
    """命令行模式：在终端实时播放 ASCII 艺术视频"""
    import os
    import sys
    import time
    import cv2

    # ANSI 终端转义码
    CLEAR_SCREEN = '\033[2J'
    HOME = '\033[H'
    HIDE_CURSOR = '\033[?25l'
    SHOW_CURSOR = '\033[?25h'
    RESET_COLOR = '\033[0m'

    print(f'{CLEAR_SCREEN}{HOME}{HIDE_CURSOR}', end='', flush=True)

    print('\033[92m' + '='*50 + '\033[0m')
    print('\033[92m   ASCII Art Warp 终端播放模式\033[0m')
    print('\033[92m' + '='*50 + '\033[0m')
    print(f'分辨率: {target_w} x {target_h}')
    print(f'彩色: {use_color}')
    print(f'加速: {accel_mode}')
    print('按 Q 或 Ctrl+C 退出\n')
    print('加载中...', end='', flush=True)

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'{SHOW_CURSOR}\033[91m错误：无法打开视频: {input_path}\033[0m')
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    frame_delay = 1.0 / fps

    # 初始化加速模式
    try:
        set_accel_mode(accel_mode)
    except:
        pass

    # 预加载字体尺寸
    try:
        from PIL import ImageFont, ImageDraw, Image
        font = ImageFont.truetype(font_path, font_size)
        total_width = sum(font.getlength(c) for c in "MWAy0g@#")
        char_width = max(1, int(total_width / 8))
    except:
        char_width = max(1, font_size // 2)
    char_height = max(1, font_size)
    scale = 256.0 / len(charset)

    # 获取终端宽度（限制最大）
    try:
        import shutil
        term_w = shutil.get_terminal_size().columns
        term_h = shutil.get_terminal_size().lines
        # 每字符宽高比校正（终端字符高度约为宽度的1.8-2倍）
        max_w = min(target_w, term_w)
        max_h = min(target_h, term_h - 2)
    except:
        max_w, max_h = target_w, target_h

    buf = []

    def frame_to_ansi(frame_bgr):
        """将一帧转换为 ANSI 彩色 ASCII 字符串"""
        if max_w != target_w or max_h != target_h:
            frame_bgr = cv2.resize(frame_bgr, (max_w, max_h))

        if use_color:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_rgb = np.stack([gray, gray, gray], axis=-1)

        lines = []
        for y in range(gray.shape[0]):
            line_chars = []
            for x in range(gray.shape[1]):
                g = gray[y, x]
                char_idx = min(int(g / scale), len(charset) - 1)
                ch = charset[char_idx]
                if use_color:
                    r, g_c, b = frame_rgb[y, x]
                    # ANSI 256 色
                    # 正确量化: 0-255 → 0-5 (ANSI 216色色码 16-231)
                    cr = r // 51
                    cg = g_c // 51
                    cb = b // 51
                    color_code = 16 + cr * 36 + cg * 6 + cb
                    line_chars.append(f'\033[38;5;{color_code}m{ch}')
                else:
                    line_chars.append(ch)
            lines.append(''.join(line_chars))

        return '\n'.join(lines)

    try:
        idx = 0
        start_time = time.time()
        last_status = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 限制渲染区域
            frame_small = cv2.resize(frame, (max_w, max_h))
            frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # 构建 ANSI 输出
            lines = []
            for y in range(frame_gray.shape[0]):
                line_chars = []
                for x in range(frame_gray.shape[1]):
                    g = frame_gray[y, x]
                    char_idx = min(int(g / scale), len(charset) - 1)
                    ch = charset[char_idx]
                    if use_color:
                        r, g_c, b = frame_rgb[y, x]
                        # 正确量化: 0-255 → 0-5 (ANSI 216色色码 16-231)
                        cr = r // 51
                        cg = g_c // 51
                        cb = b // 51
                        color_code = 16 + cr * 36 + cg * 6 + cb
                        line_chars.append(f'\033[38;5;{color_code}m{ch}')
                    else:
                        line_chars.append(ch)
                lines.append(''.join(line_chars))

            ascii_output = '\n'.join(lines)

            # 打印到终端
            print(f'{HOME}{RESET_COLOR}{ascii_output}', end='', flush=True)

            # 状态信息
            elapsed = time.time() - last_status
            if elapsed >= 1.0:
                current_fps = idx / (time.time() - start_time)
                print(f'\033[K\033[90m帧: {idx+1}/{total} | 播放: {current_fps:.1f} FPS | 退出: Q\033[0m', 
                      end='', flush=True)
                last_status = time.time()

            idx += 1

            # 控制播放速度
            time.sleep(frame_delay)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f'\n{SHOW_CURSOR}{RESET_COLOR}\033[92m播放结束，共 {idx} 帧\033[0m')
        return 0


def run_cli(args):
    """命令行模式：直接在终端播放/转换"""
    import os
    import sys
    import time

    print('\033[92m' + '='*50 + '\033[0m')
    print('\033[92m   ASCII Art Warp 命令行模式\033[0m')
    print('\033[92m' + '='*50 + '\033[0m\n')

    # 检查文件
    if not os.path.exists(args.input):
        print(f'\033[91m错误：输入文件不存在: {args.input}\033[0m')
        return 1

    # 确认输出路径
    output_path = args.output or args.input.rsplit('.', 1)[0] + '_ascii.mkv'
    print(f'输入: {args.input}')
    print(f'输出: {output_path}')
    print(f'分辨率: {args.width} x {args.height}')
    print(f'彩色: {args.color}')
    print(f'压缩: {args.compress}')
    print(f'加速: {args.accel}')
    print()

    # 开始转换
    start_time = time.time()
    frame_count = [0]
    last_update = [time.time()]

    def progress(current, total, eta_str):
        frame_count[0] = current
        elapsed = time.time() - last_update[0]
        if elapsed >= 1.0:
            pct = min(100, current * 100 // max(1, total))
            print(f'\r进度: {current}/{total} 帧 | {pct}% | ETA {eta_str}', end='', flush=True)
            last_update[0] = time.time()

    try:
        result = video_to_mkv_multiprocess(
            input_path=args.input,
            output_path=output_path,
            target_w=args.width,
            target_h=args.height,
            charset=' .:-=+*#%@',
            font_path=args.font,
            font_size=args.font_size,
            bg_color_rgb=(0, 0, 0),
            use_color=args.color,
            max_workers=args.max_workers,
            compress=args.compress,
            crf=args.crf,
            accel_mode=args.accel,
            gpu_id=args.gpu_id,
            reduce_video=args.reduce_video,
            progress_callback=progress
        )
        elapsed = time.time() - start_time
        print(f'\n\n\033[92m转换完成！\033[0m')
        print(f'总帧数: {result}')
        print(f'耗时: {elapsed:.1f} 秒')
        print(f'平均: {result/elapsed:.1f} FPS')
        print(f'输出: {output_path}')
        return 0
    except Exception as e:
        print(f'\n\033[91m错误: {e}\033[0m')
        import traceback
        traceback.print_exc()
        return 1


class AsciiArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("字符画转换器（多进程 + GPU加速）")
        self.root.geometry("1000x960")

        self.file_path = None
        self.processing = False
        self.original_aspect = 1.0
        self.original_width_px = 0
        self.original_height_px = 0

        system_font = get_system_monospace_font()
        self.font_path_var = tk.StringVar(value=system_font if system_font else "Consolas.ttf")
        self.bg_color_rgb = (0, 0, 0)

        self.max_cpu_cores = mp.cpu_count()
        self.cpu_cores_var = tk.IntVar(value=min(4, self.max_cpu_cores))

        # GPU 相关
        self.gpu_devices = _TORCH_DEVICES
        self.accel_options = _ACCEL_OPTIONS
        self.accel_mode = _ACCEL_CURRENT
        self.selected_gpu_id = 0

        self.create_widgets()

    def create_widgets(self):
        # ---- 硬件加速面板 ----
        fgpu = tk.Frame(self.root, bg="#0d1117", bd=1, relief="solid")
        fgpu.pack(fill="x", padx=10, pady=(10, 0))

        hdr = tk.Frame(fgpu, bg="#0d1117")
        hdr.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(hdr, text="🖥  硬件加速", fg="#e6edf3", bg="#0d1117",
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        if self.gpu_devices:
            gpu_info = "  |  ".join(f"[{d[0]}] {d[1]} ({d[2]}GB)" for d in self.gpu_devices)
            tk.Label(hdr, text=f"检测到 {len(self.gpu_devices)} 台 GPU: {gpu_info}",
                     fg="#58a6ff", bg="#0d1117", font=("Consolas", 8)).pack(side="right")
        else:
            tk.Label(hdr, text="未检测到 CUDA GPU", fg="#f0883e", bg="#0d1117",
                     font=("Consolas", 8)).pack(side="right")

        row1 = tk.Frame(fgpu, bg="#0d1117")
        row1.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(row1, text="加速后端:", fg="#c9d1d9", bg="#0d1117").pack(side="left")

        backend_vals = [o[1] for o in self.accel_options]
        backend_map = {o[1]: (o[0], o[2]) for o in self.accel_options}

        self.accel_var = tk.StringVar(value=backend_vals[0] if backend_vals else "")
        self.accel_combo = ttk.Combobox(row1, textvariable=self.accel_var,
                                        values=backend_vals, width=30, state="readonly")
        self.accel_combo.pack(side="left", padx=(0, 10))
        self.accel_combo.bind('<<ComboboxSelected>>', self._on_accel_changed)

        first_desc = backend_map.get(backend_vals[0], ("", ""))[1]
        self.accel_desc_var = tk.StringVar(value=first_desc)
        tk.Label(row1, textvariable=self.accel_desc_var, fg="#8b949e", bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side="left")

        row2 = tk.Frame(fgpu, bg="#0d1117")
        row2.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(row2, text="使用显卡:", fg="#c9d1d9", bg="#0d1117").pack(side="left")

        if self.gpu_devices:
            gpu_vals = [f"[{d[0]}] {d[1]} ({d[2]}GB)" for d in self.gpu_devices]
            self.gpu_var = tk.StringVar(value=gpu_vals[0])
            self.gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var,
                                          values=gpu_vals, width=30, state="readonly")
            self.gpu_combo.pack(side="left", padx=(0, 10))
            self.gpu_combo.bind('<<ComboboxSelected>>', self._on_gpu_changed)

            if len(self.gpu_devices) == 1:
                tk.Label(row2, text="(单卡模式，切换到 DataParallel 可用多卡)",
                         fg="#6e7681", bg="#0d1117", font=("Segoe UI", 8)).pack(side="left")
            else:
                tk.Label(row2, text=f"(共 {len(self.gpu_devices)} 卡，DataParallel 自动轮询)",
                         fg="#58a6ff", bg="#0d1117", font=("Segoe UI", 8)).pack(side="left")
        else:
            tk.Label(row2, text="无可用 GPU（将使用 CPU 加速）",
                     fg="#f0883e", bg="#0d1117", font=("Segoe UI", 9)).pack(side="left")

        self.gpu_status = tk.Label(fgpu, text=f"当前: {_ACCEL_MODE}",
                                   fg="#3fb950", bg="#0d1117", font=("Consolas", 9))
        self.gpu_status.pack(anchor="w", padx=10, pady=(0, 6))

        # ---- 文件选择 ----
        ffile = tk.LabelFrame(self.root, text="1. 选择媒体文件", padx=5, pady=5)
        ffile.pack(fill="x", padx=10, pady=5)
        tk.Button(ffile, text="打开图片/视频", command=self.select_file).pack(side="left", padx=5)
        self.lbl_file = tk.Label(ffile, text="未选择", fg="gray")
        self.lbl_file.pack(side="left", padx=5)
        self.btn_preview = tk.Button(ffile, text="预览悬浮窗", command=self.open_preview, state="disabled")
        self.btn_preview.pack(side="left", padx=20)

        # ---- 分辨率 ----
        fres = tk.LabelFrame(self.root, text="2. 输出分辨率（字符数）", padx=5, pady=5)
        fres.pack(fill="x", padx=10, pady=5)
        tk.Label(fres, text="宽度（列数）:").grid(row=0, column=0, sticky="w", padx=5)
        self.width_var = tk.IntVar(value=80)
        tk.Spinbox(fres, from_=20, to=500, textvariable=self.width_var, width=6,
                   command=self.on_width_change).grid(row=0, column=1, sticky="w")
        tk.Label(fres, text="高度（行数）:").grid(row=0, column=2, sticky="w", padx=5)
        self.height_var = tk.IntVar(value=45)
        tk.Spinbox(fres, from_=10, to=300, textvariable=self.height_var, width=6,
                   command=self.on_height_change).grid(row=0, column=3, sticky="w")
        self.lock_aspect = tk.BooleanVar(value=True)
        tk.Checkbutton(fres, text="锁定宽高比", variable=self.lock_aspect).grid(
            row=0, column=4, columnspan=2, sticky="w", padx=10)
        self.lbl_suggestion = tk.Label(fres, text="建议分辨率：--", fg="gray")
        self.lbl_suggestion.grid(row=1, column=0, columnspan=6, sticky="w", padx=5, pady=2)

        # ---- 字符样式与性能 ----
        fpar = tk.LabelFrame(self.root, text="3. 字符样式与性能", padx=5, pady=5)
        fpar.pack(fill="x", padx=10, pady=5)

        r = 0
        tk.Label(fpar, text="字符集:").grid(row=r, column=0, sticky="w", padx=5)
        self.charset_var = tk.StringVar(value=" .:-=+*#%@")
        ttk.Combobox(fpar, textvariable=self.charset_var, values=(
            " .:-=+*#%@",
            " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
            "01",
            "@%#*+=-:. "
        ), width=30).grid(row=r, column=1, sticky="w")
        r += 1

        tk.Label(fpar, text="字体:").grid(row=r, column=0, sticky="w", padx=5)
        tk.Entry(fpar, textvariable=self.font_path_var, width=20).grid(row=r, column=1, sticky="w")
        tk.Label(fpar, text="(等宽字体)").grid(row=r, column=2, columnspan=2, sticky="w")
        r += 1

        tk.Label(fpar, text="字号:").grid(row=r, column=0, sticky="w", padx=5)
        self.fontsize_var = tk.IntVar(value=16)
        tk.Spinbox(fpar, from_=8, to=48, textvariable=self.fontsize_var, width=6).grid(row=r, column=1, sticky="w")
        r += 1

        self.use_color_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fpar, text="彩色输出（保留原图颜色）",
                       variable=self.use_color_var).grid(row=r, column=0, columnspan=2, sticky="w", padx=5)
        r += 1

        tk.Label(fpar, text="CPU 进程数:").grid(row=r, column=0, sticky="w", padx=5)
        self.cpu_spin = tk.Spinbox(fpar, from_=1, to=self.max_cpu_cores,
                                    textvariable=self.cpu_cores_var, width=6)
        self.cpu_spin.grid(row=r, column=1, sticky="w")
        tk.Label(fpar, text=f"(本机最大 {self.max_cpu_cores} 核心)").grid(
            row=r, column=2, columnspan=2, sticky="w")
        r += 1

        tk.Label(fpar, text="内存缓冲帧数:").grid(row=r, column=0, sticky="w", padx=5)
        self.buffer_size_var = tk.IntVar(value=60)
        tk.Spinbox(fpar, from_=5, to=200, textvariable=self.buffer_size_var, width=6).grid(row=r, column=1, sticky="w")
        tk.Label(fpar, text="(调大可提升 GPU 利用率)").grid(row=r, column=2, columnspan=2, sticky="w")
        r += 1

        self.keep_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fpar, text="保留原视频音频（需要 ffmpeg）",
                       variable=self.keep_audio_var).grid(row=r, column=0, columnspan=4, sticky="w", padx=5)
        r += 1

        self.reduce_video_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fpar, text="预压缩视频（缩小到目标分辨率，减少处理量）",
                       variable=self.reduce_video_var).grid(row=r, column=0, columnspan=4, sticky="w", padx=5)
        r += 1

        tk.Label(fpar, text="输出压缩强度:").grid(row=r, column=0, sticky="w", padx=5)
        self.strength_var = tk.IntVar(value=5)
        strength_scale = tk.Scale(fpar, from_=0, to=10, orient="horizontal", variable=self.strength_var,
                                  length=200, resolution=1)
        strength_scale.grid(row=r, column=1, columnspan=2, sticky="w")
        self.lbl_strength = tk.Label(fpar, text="5 (平衡)", width=15)
        self.lbl_strength.grid(row=r, column=3, sticky="w")
        def _on_strength(v):
            v = int(v)
            if v == 0:
                self.lbl_strength.config(text="0 (不压缩)")
            elif v <= 3:
                self.lbl_strength.config(text=f"{v} (轻压缩)")
            elif v <= 7:
                self.lbl_strength.config(text=f"{v} (平衡)")
            else:
                self.lbl_strength.config(text=f"{v} (强压缩)")
        strength_scale.config(command=_on_strength)
        r += 1

        # ---- 转换按钮 & 进度 ----
        tk.Frame(self.root).pack(fill="x", padx=10, pady=5)
        self.btn_convert = tk.Button(self.root, text="开始转换", command=self.start_conversion,
                                      bg="lightgreen", font=("Arial", 11))
        self.btn_convert.pack(pady=5)


        fprog = tk.Frame(self.root)
        fprog.pack(fill="x", padx=10, pady=5)
        self.progress = ttk.Progressbar(fprog, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(side="left", padx=5)
        self.lbl_eta = tk.Label(fprog, text="ETA: --:--:--", width=15)
        self.lbl_eta.pack(side="left", padx=10)
        self.lbl_status = tk.Label(self.root, text="就绪", fg="blue")
        self.lbl_status.pack(pady=5)

    def _on_accel_changed(self, event=None):
        label = self.accel_var.get()
        for o in self.accel_options:
            if o[1] == label:
                self.accel_mode = o[0]
                set_accel_mode(o[0])
                self.accel_desc_var.set(o[2])
                self.gpu_status.config(text=f"当前: {_ACCEL_MODE}")
                if "CUDA" in _ACCEL_MODE:
                    self.gpu_status.config(fg="#3fb950")
                elif "CPU" in _ACCEL_MODE or "Numba" in _ACCEL_MODE:
                    self.gpu_status.config(fg="#d29922")
                else:
                    self.gpu_status.config(fg="#8b949e")
                break

    def _on_gpu_changed(self, event=None):
        try:
            sel = self.gpu_var.get()
            self.selected_gpu_id = int(sel.split(']')[0].split('[')[1])
        except:
            self.selected_gpu_id = 0

    def select_file(self):
        path = filedialog.askopenfilename(title="选择图片或视频", filetypes=[
            ("所有支持", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.flv")])
        if path:
            self.file_path = path
            self.lbl_file.config(text=os.path.basename(path), fg="black")
            self.btn_preview.config(state="normal")
            self.get_original_dimensions(path)
            self.update_suggestion()

    def get_original_dimensions(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                with Image.open(filepath) as img:
                    self.original_width_px, self.original_height_px = img.size
            else:
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    self.original_width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.original_height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                else:
                    self.original_width_px, self.original_height_px = 1920, 1080
            self.original_aspect = self.original_width_px / self.original_height_px
        except:
            self.original_width_px, self.original_height_px = 1920, 1080
            self.original_aspect = 16/9

    def update_suggestion(self):
        if self.original_aspect > 0:
            w = self.width_var.get()
            h = int(w / self.original_aspect)
            h = max(10, min(300, h))
            self.lbl_suggestion.config(text=f"建议分辨率（保持比例）：{w} x {h}")
        else:
            self.lbl_suggestion.config(text="建议分辨率：请先选择文件")

    def on_width_change(self):
        if self.lock_aspect.get() and self.original_aspect > 0:
            w = self.width_var.get()
            h = int(w / self.original_aspect)
            h = max(10, min(300, h))
            self.height_var.set(h)
        self.update_suggestion()

    def on_height_change(self):
        if self.lock_aspect.get() and self.original_aspect > 0:
            h = self.height_var.get()
            w = int(h * self.original_aspect)
            w = max(20, min(500, w))
            self.width_var.set(w)
        self.update_suggestion()

    def open_preview(self):
        if not self.file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return
        charset = self.charset_var.get()
        if len(charset) < 2:
            messagebox.showwarning("警告", "字符集至少需要 2 个字符")
            return
        PreviewWindow(self.root, self.file_path, self.width_var.get(), self.height_var.get(),
                      charset, self.font_path_var.get(), self.fontsize_var.get(),
                      self.bg_color_rgb, self.use_color_var.get())

    def start_conversion(self):
        if not self.file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return
        charset = self.charset_var.get()
        if len(charset) < 2:
            messagebox.showwarning("警告", "字符集至少需要 2 个字符")
            return
        if self.processing:
            return

        ext = os.path.splitext(self.file_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']

        if is_video:
            output_path = filedialog.asksaveasfilename(defaultextension=".mkv", filetypes=[("MKV 视频", "*.mkv")])
        else:
            output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 图片", "*.png")])

        if not output_path:
            return

        self.processing = True
        self.btn_convert.config(state="disabled")
        self.progress["value"] = 0
        self.lbl_eta.config(text="ETA: --:--:--")
        self.lbl_status.config(text="转换中...")

        target_w = self.width_var.get()
        target_h = self.height_var.get()
        max_workers = min(self.cpu_cores_var.get(), self.max_cpu_cores)
        buffer_size = self.buffer_size_var.get()
        font_path_abs = os.path.abspath(self.font_path_var.get())

        def update_progress(current, total, eta_str):
            self.root.after(0, lambda: self.progress.config(maximum=total, value=current))
            self.root.after(0, lambda: self.lbl_eta.config(text=f"ETA: {eta_str}"))
            self.root.after(0, lambda: self.lbl_status.config(text=f"处理 {current}/{total}"))
            self.root.after(0, lambda: self.root.update_idletasks())

        def update_compress_progress(current_sec, total_sec):
            pct = min(int(current_sec / total_sec * 100), 99) if total_sec > 0 else 0
            self.root.after(0, lambda crs=current_sec, trs=total_sec, p=pct: [
                self.progress.configure(maximum=100, value=p),
                self.lbl_eta.config(text=""),
                self.lbl_status.config(text=f"压缩中... {crs:.0f}s/{trs:.0f}s ({p}%)")
            ])

        def worker():
            temp_video = None
            try:
                if is_video:
                    temp_dir = os.path.dirname(output_path) or "."
                    temp_video = os.path.join(temp_dir, "_temp_no_audio.mp4")
                    total = video_to_mkv_multiprocess(
                        input_path=self.file_path, output_path=temp_video,
                        target_w=target_w, target_h=target_h, charset=charset,
                        font_path=font_path_abs, font_size=self.fontsize_var.get(),
                        bg_color_rgb=self.bg_color_rgb, use_color=self.use_color_var.get(),
                        max_workers=max_workers, buffer_size=buffer_size,
                        progress_callback=update_progress,
                        accel_mode=self.accel_mode, gpu_id=self.selected_gpu_id,
                        reduce_video=self.reduce_video_var.get(),
                        compress=self.strength_var.get() > 0,
                        crf=16 + self.strength_var.get() * 2 if self.strength_var.get() > 0 else 23,
                        compress_progress_callback=update_compress_progress)

                    if self.keep_audio_var.get():
                        self.root.after(0, lambda: self.lbl_status.config(text="正在合并音频..."))
                        ok, msg = merge_audio(temp_video, self.file_path, output_path)
                        if ok:
                            try:
                                os.remove(temp_video)
                            except:
                                pass
                            self.root.after(0, lambda: messagebox.showinfo("完成",
                                f"视频转换完成！共 {total} 帧\n已保留原音频\n保存至：{output_path}"))
                        else:
                            if os.path.exists(temp_video):
                                try:
                                    os.rename(temp_video, output_path)
                                except:
                                    pass
                            if "没有音轨" in msg:
                                self.root.after(0, lambda: messagebox.showinfo("完成",
                                    f"视频转换完成！共 {total} 帧\n原视频无音轨\n保存至：{output_path}"))
                            elif "未安装" in msg:
                                self.root.after(0, lambda: messagebox.showwarning("警告",
                                    f"音频合并失败：{msg}"))
                            else:
                                self.root.after(0, lambda: messagebox.showwarning("警告",
                                    f"音频合并失败：{msg}\n输出视频无音频"))
                    else:
                        if os.path.exists(temp_video):
                            os.rename(temp_video, output_path)
                        self.root.after(0, lambda: messagebox.showinfo("完成",
                            f"视频转换完成！共 {total} 帧\n保存至：{output_path}"))
                else:
                    image_to_image(input_path=self.file_path, output_path=output_path,
                                   target_w=target_w, target_h=target_h, charset=charset,
                                   font_path=font_path_abs, font_size=self.fontsize_var.get(),
                                   bg_color_rgb=self.bg_color_rgb, use_color=self.use_color_var.get())
                    self.root.after(0, lambda: messagebox.showinfo("完成",
                        f"图片转换完成\n保存至：{output_path}"))

                self.root.after(0, lambda: self.lbl_status.config(text="转换完成"))
                self.root.after(0, lambda: self.lbl_eta.config(text="ETA: 00:00:00"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
                self.root.after(0, lambda: self.lbl_status.config(text="转换失败"))
                if temp_video and os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except:
                        pass
            finally:
                self.root.after(0, lambda: setattr(self, 'processing', False))
                self.root.after(0, lambda: self.btn_convert.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='ASCII Art Warp - 视频转ASCII艺术',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # GUI 模式（默认）
  python ascii_art_warp_final.py

  # 命令行转换
  python ascii_art_warp_final.py --cli --input video.mp4 --output ascii.mkv

  # 命令行转换（高分辨率）
  python ascii_art_warp_final.py --cli -i video.mp4 -o ascii.mkv -w 120 -h 60 --color

  # 在终端播放ASCII艺术视频
  python ascii_art_warp_final.py --watch -i video.mp4

  # 在终端播放（高分辨率）
  python ascii_art_warp_final.py --watch -i video.mp4 -w 120 -h 60 --color


        '''
    )

    parser.add_argument('--cli', action='store_true', help='使用命令行模式（否则启动GUI）')

    parser.add_argument('--watch', action='store_true', help='在终端实时播放ASCII艺术视频')

    # 转换参数
    parser.add_argument('-i', '--input', type=str, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, help='输出视频路径')
    parser.add_argument('-w', '--width', type=int, default=80, help='输出宽度（字符数，默认80）')
    parser.add_argument('-H', '--height', type=int, default=45, help='输出高度（字符数，默认45）')
    parser.add_argument('--charset', type=str, default=' .:-=+*#%@', help='字符集')
    parser.add_argument('--font', type=str, default='consola.ttf', help='字体文件路径')
    parser.add_argument('--font-size', type=int, default=16, help='字体大小（默认16）')
    parser.add_argument('--color', action='store_true', default=True, help='使用彩色模式')
    parser.add_argument('--no-color', dest='color', action='store_false', help='禁用彩色模式（灰度）')
    parser.add_argument('--compress', action='store_true', default=True, help='压缩输出视频')
    parser.add_argument('--no-compress', dest='compress', action='store_false', help='不压缩输出视频')
    parser.add_argument('--crf', type=int, default=23, help='视频质量 0-51（越小越好，默认23）')
    parser.add_argument('--preset', type=str, default='medium', help='编码速度 preset')
    parser.add_argument('--accel', type=str, default='torch_cuda_single',
                       choices=['torch_cuda_single', 'torch_cuda_dataparallel', 'torch_cpu', 'numba_parallel', 'numpy_vectorized'],
                       help='加速模式')
    parser.add_argument('--max-workers', type=int, default=4, help='CPU worker数量')
    parser.add_argument('--reduce-video', action='store_true', default=True, help='预处理缩小视频，减少内存占用和进程间传输带宽')
    parser.add_argument('--no-reduce-video', dest='reduce_video', action='store_false', help='不缩小视频，保持原始帧尺寸')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU设备ID')

    args = parser.parse_args()



    # 终端播放模式
    if args.watch:
        if not args.input:
            print('\033[91m错误：--watch 模式需要指定输入文件 -i\033[0m')
            sys.exit(1)
        sys.exit(watch_video(
            input_path=args.input,
            target_w=args.width,
            target_h=args.height,
            charset=' .:-=+*#%@',
            font_path=args.font,
            font_size=args.font_size,
            use_color=args.color,
            accel_mode=args.accel,
            gpu_id=args.gpu_id,
            reduce_video=args.reduce_video
        ))

    # CLI 模式
    if args.cli:
        sys.exit(run_cli(args))

    # GUI 模式
    root = tk.Tk()
    app = AsciiArtApp(root)
    root.mainloop()