@echo off
echo ��ʼ�������ú�������װ...

REM ��� Python �Ƿ�װ
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ����δ�ҵ� Python�����Ȱ�װPython 3.12.2 ����߰汾 ��������ӵ� PATH��
    echo �����Դ� https://www.python.org/downloads/ ���ء�
    pause
    exit /b
)
echo Python �Ѽ�⵽��

REM �������⻷��
echo ���ڴ������⻷�� .venv...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ���󣺴������⻷��ʧ�ܡ�
    pause
    exit /b
)
echo ���⻷�� .venv �����ɹ���

REM �������⻷��
echo ���ڼ������⻷��...
call .\.venv\Scripts\activate.bat
REM ��ͬһ���ű��У�����󣬺����� pip �� python ����ͻ��ڸ����⻷����ִ��

REM ��ʾ�û�ѡ�� PyTorch �汾
echo.
echo ===============================================================================
echo ���� PyTorch ��װ (��Ҫ��)��
echo ===============================================================================
echo ������� NVIDIA GPU ��ϣ��ʹ�� CUDA ���٣�ǿ�ҽ�������
echo   1. ���� PyTorch ���� (https://pytorch.org/get-started/locally/)
echo   2. ��ȡ�ʺ��� CUDA �汾�� PyTorch ��װ���
echo   3. �ڡ��µ������д������ȼ�������⻷��(.venv\Scripts\activate)����
echo      Ȼ���ֶ�ִ�С��� PyTorch ��װ���
echo   4. �ֶ���װ PyTorch GPU �汾��֮�󡿣��ٻص����˴��ڡ��� 'n' �����Զ���װ��
echo.
echo �������ȷ����ֻ��ʹ�� CPU���������ֶ���װ GPU �� PyTorch��
echo �ű����Գ��԰�װһ��ͨ�õ� PyTorch (ͨ����CPU��)������������
echo ===============================================================================
echo.

:pytorch_choice_prompt
set "install_pytorch_choice="
set /p install_pytorch_choice="���Ƿ����ֶ���װ PyTorch GPU �汾��(y/n������ 'y' �����Զ���װ��'n' �ýű����԰�װ): "

if /i "%install_pytorch_choice%"=="y" (
    echo ��ѡ�������ֶ���װ������ PyTorch �Զ���װ��
) else if /i "%install_pytorch_choice%"=="n" (
    echo ���ڳ��԰�װ PyTorch (�������CPU�汾��GPU�û���ȷ�����ֶ���װGPU��)...
    pip install torch torchvision torchaudio --no-cache-dir
    if %errorlevel% neq 0 (
        echo ���棺PyTorch �Զ���װʧ�ܻ��������⡣�����ҪGPU֧�֣�������ֶ���װ��
        pause
    ) else (
        echo PyTorch �Զ���װ������ɡ�
    )
) else (
    echo ��Ч���룬������ y �� n��
    goto pytorch_choice_prompt
)

REM ��װ��������
echo.
echo ���ڰ�װ�������� (���� requirements.txt)...
pip install -r requirements.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo ���󣺴� requirements.txt ��װ����ʧ�ܡ�
    echo ���� requirements.txt �ļ��Ƿ�����Ҹ�ʽ��ȷ���Լ������������ӡ�
    pause
    exit /b
)
echo ����������װ��ɡ�

echo.
echo ===============================================================================
echo ��Ҫ��ʾ��FFmpeg ��װ (����)
echo ===============================================================================
echo ����Ŀ������Ҫ FFmpeg��
echo ��ȷ�����Ѵ� https://ffmpeg.org/download.html (�Ƽ� gyan.dev ����)
echo ���� FFmpeg���������е� 'bin' Ŀ¼·����ӵ�ϵͳ�� PATH ���������С�
echo.
echo ��μ��FFmpeg�Ƿ����óɹ���
echo   ��һ���µ������д��ڣ����� 'ffmpeg -version'�������ʾ�汾��Ϣ���ʾ�ɹ���
echo ===============================================================================
echo.
echo ����������İ�װ������ִ����ϡ�
echo.
echo ��һ����
echo   1. �������δ���� FFmpeg�����������á�
echo   2. ������ɺ�������˫������ 'launcher.bat' ������Ӧ�ó���
echo.
pause
exit /b