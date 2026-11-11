@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."

echo ============================================
echo  本地再铸 V0.1  Windows 打包（PyInstaller onedir）
echo ============================================
echo.
echo 请在已经成功运行 scripts\install_windows.bat 的环境里执行本脚本。
echo 本脚本不会把大模型权重打进安装包；模型仍在首次运行时按需下载并校验。
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv。请先运行 scripts\install_windows.bat。
    exit /b 1
)
call ".venv\Scripts\activate.bat"

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo 正在安装 PyInstaller ...
    pip install "pyinstaller>=6.6"
    if errorlevel 1 (
        echo [错误] 安装 PyInstaller 失败。
        exit /b 1
    )
)

if not exist "dist" mkdir dist
if not exist "build" mkdir build

echo 正在打包（onedir）...
pyinstaller --noconfirm recast.spec
if errorlevel 1 (
    echo [错误] 打包失败。
    exit /b 1
)

echo.
echo 正在处理 ffmpeg 固定路径 ...
if exist "ffmpeg\ffmpeg.exe" (
    if not exist "dist\本地再铸\ffmpeg" mkdir "dist\本地再铸\ffmpeg"
    copy /Y "ffmpeg\ffmpeg.exe" "dist\本地再铸\ffmpeg\ffmpeg.exe" >nul
    if exist "ffmpeg\ffprobe.exe" copy /Y "ffmpeg\ffprobe.exe" "dist\本地再铸\ffmpeg\ffprobe.exe" >nul
    echo 已把项目内 ffmpeg\ 复制到输出目录。
) else (
    echo [提示] 未在项目根目录发现 ffmpeg\ffmpeg.exe。
    echo         请把 ffmpeg.exe 与 ffprobe.exe 放到:
    echo           dist\本地再铸\ffmpeg\
    echo         或继续依赖系统 PATH。
)

echo.
echo 打包完成。输出目录:
echo   dist\本地再铸\本地再铸.exe
echo.
echo 使用前请在目标电脑安装 NVIDIA 驱动。首次运行「人物替换」仍会下载 inswapper 等权重
echo 到 models\ 或用户目录，不会随本包预置大文件。
echo 请使用 Python 3.11 环境打包，不要用 3.13。
endlocal
