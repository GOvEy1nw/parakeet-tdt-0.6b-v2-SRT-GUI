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


REM ������⻷���Ƿ����
if not exist .\.venv\Scripts\activate.bat (
	REM �������⻷��
	echo ���ڴ������⻷�� .venv...
	python -m venv .venv
	if %errorlevel% neq 0 (
	echo ���󣺴������⻷��ʧ�ܡ�
	pause
	exit /b
	)
	echo ���⻷�� .venv �����ɹ���
)

REM �������⻷��
call .\.venv\Scripts\activate.bat
echo ���⻷���Ѽ��

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


set /p choice="�Ƿ�װtorch (cpu�汾) ����������(y/n): "
if /i "%choice%"=="y" (
	echo ���ڰ�װtorch��������

	pip install -r requirements.txt

	if %errorlevel% neq 0 (
		echo ���󣺰�װ����ʧ�ܡ�
		pause
		exit /b
	)
	echo ������װ�ɹ���
) else (
		pause
		exit /b
)
 

echo requirements������װ��ɡ�

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





