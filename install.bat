@echo off
chcp 65001 >nul

setlocal enabledelayedexpansion

:: ============================================================
:: ASCII Art Warp - 一键安装脚本
:: ============================================================
:: 自动安装 Python 依赖和 ffmpeg
:: ============================================================

set SCRIPT_DIR=%~dp0
set SCRIPT_FILE=ascii_art_warp_final.py
set REPO_URL=https://github.com/qing-lin-z/ascii-art-warp

:: === 检查 Python ===
call :check_python
if %ERRORLEVEL% neq 0 (
    echo [错误] 未检测到 Python
    echo 请先安装 Python 3.10 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    echo.
    echo 安装时记得勾选 "Add Python to PATH"
    pause
    exit /b 1
)

:: === 检查 pip ===
python -m pip --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] pip 不可用，尝试安装...
    python -m ensurepip --upgrade
    if !ERRORLEVEL! neq 0 (
        pause
        exit /b 1
    )
)

echo.
echo =============================================
echo    ASCII Art Warp - 一键安装
echo =============================================
echo.

:: ============================================================
:: 步骤 1: 安装核心依赖
:: ============================================================
echo [1/4] 安装核心依赖 (numpy, opencv, pillow, pygame)...
python -m pip install numpy opencv-python pillow pygame --user --quiet
if %ERRORLEVEL% neq 0 (
    echo [错误] 安装失败，请检查网络连接
    pause
    exit /b 1
)
echo   完成
echo.

:: ============================================================
:: 步骤 2: 检测 GPU 并安装 PyTorch
:: ============================================================
echo [2/4] 检测 GPU 并安装 PyTorch...

where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   检测到 NVIDIA 显卡，尝试安装 CUDA 版 PyTorch...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --user --quiet
    if !ERRORLEVEL! equ 0 (
        echo   CUDA 版安装成功！GPU 加速可用
    ) else (
        echo   CUDA 版安装失败，降级到 CPU 版...
        python -m pip install torch torchvision --user --quiet
    )
) else (
    echo   未检测到 NVIDIA 显卡，安装 CPU 版 PyTorch...
    python -m pip install torch torchvision --user --quiet
)
echo   完成
echo.

:: ============================================================
:: 步骤 3: 安装可选加速库
:: ============================================================
echo [3/4] 安装可选加速库 (numba, moviepy)...
python -m pip install numba moviepy --user --quiet 2>nul
echo   完成
echo.

:: ============================================================
:: 步骤 4: 检测 ffmpeg（视频编码必需）
:: ============================================================
echo [4/4] 检测 ffmpeg...

where ffmpeg >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%a in ('ffmpeg -version 2^>nul ^| findstr /i "ffmpeg version"') do set FFMPEG_VER=%%a
    echo   已安装: !FFMPEG_VER!
) else (
    echo   [警告] 未检测到 ffmpeg
    echo   视频编码功能需要 ffmpeg
    echo.
    echo   下载地址: https://ffmpeg.org/download.html
    echo   或使用包管理器: winget install ffmpeg
    echo.
    set FFMPEG_MISSING=1
)
echo   完成
echo.

:: ============================================================
:: 验证安装结果
:: ============================================================
echo.
echo =============================================
echo   安装摘要
echo =============================================

for %%p in (numpy cv2 PIL pygame torch) do (
    python -c "import %%p" >nul 2>nul && (
        python -c "import %%p; print('   [OK] %%p ' + (getattr(%%p, '__version__', '') if hasattr(%%p, '__version__') else ''))" 2>nul || echo   [OK] %%p
    ) || (
        echo   [!!] %%p 未安装
    )
)

:: GPU 检测
python -c "import torch; print('   [GPU] CUDA可用:' + str(torch.cuda.is_available()) + (' 显卡:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))" 2>nul

if defined FFMPEG_MISSING (
    echo   [!!] ffmpeg 未安装
) else (
    echo   [OK] ffmpeg
)

echo.
echo =============================================
echo   安装完成！
echo =============================================
echo.
echo 使用方法:
echo.
echo   GUI 模式:   双击运行 %SCRIPT_FILE%
echo.
echo   命令行转换: python %SCRIPT_FILE% --cli -i 视频.mp4 -o 输出.mp4
echo.
echo   终端播放:   python %SCRIPT_FILE% --watch -i 视频.mp4 --color
echo.
pause
goto :eof

:check_python
python --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    python3 --version >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        exit /b 1
    )
)
exit /b 0
