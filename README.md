
#parakeet-tdt-0.6b-v2-SRT-GUI - NeMo based video/audioSRT subtitle generation tool

The project uses the nvidia/parakeet-tdt-0.6b-v2 ASR (Auto-Voice Recognition) model to automatically generate SRT subtitle files with timetamps from video or audio files. The interface is constructed through Gradio to facilitate user uploading of files and obtaining results.

## Main function

* Extracting audio from a variety of common video formats (e. g. MP4, MKV, AVI) and generating SRT subtitles.
* Directly processing a variety of common audio formats (e.g. MP3, WAV, M4A, FLAC) to generate SRT subtitles.
* Support long video and audio input
* Supports pre-training on NVIDIA NGC Parakeet cloud model (default `nvidia/parakeet-tdt-0.6b-v2 ' ).
* Support the loading of local `.nemo ' model files of users.
* The length of sections that can be adjusted for audio processing to strike a balance between speed and context consistency.
* Auto-detect CUDA GPU, with priority given to GPU for accelerated processing; if no GPU, run on CPU (lower speed).
* User-friendly interfaces and simple operations.
* Automatically save the model and segment length configuration selected by the user.





## Environmental requirements

* Python 3.12.2 or higher to ensure compatibility with the latest NeMo library
* **FFmpeg**: for sound and video decoding and formatting. ** Must be installed and configured separately to the system PATH environment variable. **
* NVIDIA GPU (recommended for acceleration, CUDA driven). If you don't have GPU, you can run on CPU, but at very slow speed.

## Simplified installation steps (windows)

1. **Clone Repo:**
```bash
git clone https://github.com/NINIYOYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
````

2.  **Double click to open install_dependencies.bat**
It creates and activates the Python virtual environment while checking and installing dependency.
Toch's installation depends on whether you need the GPU to accelerate.
    

3. ** Installed FFmpeg:**
This project relies on FFmpeg for audio extraction and pre-processing. You need to install it separately and ensure that its executable path is added to the system's PATH environment variable.


** Windows:**
1. Download pre-edited versions (e.g. construction from "gyan.dev" or "BtbN") from the FFmpeg official web page ([https://ffmpeg.org/download.html].
2. Depressed documents.
Add the `bin ' directory path (e. g. `C:\ffmpeg\bin ') to the system environment variable `Path ' in the depressed folder.
*   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
* ** MacOS (using Homebrew):**
```bash
brew install ffmpeg
```

3. ** Double click to open launcher.bat**
If your environment and your dependency are installed, double-click on lancher.bat will run the project.




# Installation steps

1. **Cloned warehouse:**
```bash
Gymnasium, glitcone https://github.com/NINYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
````

2. ** Create and activate the Python virtual environment (recommended strongly):**
```bash
I don't know.
````
* Windows:
```bash
....venv\scripts\activate
````
? MacOS / Linux:
```bash
I'm sorry, I'm sorry.
````

3. **PyTorch installation (important: GPU user please pay particular attention! ):**
If you want to use NVIDIA GPU for accelerated processing (recommended strongly),** you must manually install a PyTorch version compatible with your CUDA environment before installing other dependencies. **
* Enter Win+R key to open the Windows system ' s running window and enter CMD into terminal input
```bash
nvidia-smi
````
CUDA Version:
* Access to [PyTorch official network installation guide page] (https://pytorch.org/get-stard/locally/).
* Select the correct installation command based on your operating system, package manager (recommended `pip ' ), computing platform (e.g. CUDA 11.8, CUDA 12.1) and Python versions.
:: For example, if you use `pip ' and your system has a CUDA 12.1 environment, you can run:
```bash
Pip3 install tooltochvision Tochaudio-index-url https://download.pytorch.org/whl/cu121
````
If you skip this step, or your system does not have NVIDIA GPU, the `nemo_toolkit ' of subsequent installation may by default install a PyTorch version that only supports CPU.

4.** Reliance on installation projects:**
Runs:
```bash
Pip install-r reviews.txt
````
(See below for examples of `requirements.txt ' )

5. ** Installed FFmpeg:**
This project relies on FFmpeg for audio extraction and pre-processing. You need to install it separately and ensure that its executable path is added to the system's PATH environment variable.
** Windows:**
1. Download pre-edited versions (e.g. construction from "gyan.dev" or "BtbN") from the FFmpeg official web page ([https://ffmpeg.org/download.html].
2. Depressed documents.
Add the `bin ' directory path (e. g. `C:\ffmpeg\bin ') to the system environment variable `Path ' in the depressed folder.
* ** Linux (Ubuntu/Debian et al.):**
```bash
Sodo apt update & sub apt install ffmpeg
````
* ** MacOS (using Homebrew):**
```bash
I'm sorry.
````
6. ** Double click to open launcher.bat**
If your environment and your dependency are installed, double-click on lancher.bat will run the project.



# Loading models from local
** If you want to load the nvidia/parakeet-tdt-0.6b-v2 model locally, enter the model's local path on the local model path (.nemo file) at initial start**
** e.g. C: \Users\models -- nvidia -- paraket-tdt-0.6b-v2\snapsshots\30c5e6f557f6ba26e5819a99ed2e86f670186b43f\parakeet-tdt-0.6b-v2.nemo**



# Interface presentation
! [Interface Presentation] (./RESADME.assets/1.png)





Use method

Ensure that you have completed the environmental configuration and dependency installation in accordance with the above steps.

In the root directory, run the main script:

Start in two ways.
1. ** Run launcher.bat start**

2. ** Start at terminal (entry into virtual environment)**
```bash
I'm sorry, Python Main.
````



A local URL (usually http://127.0.0.1:7860 or similar address) is printed at the terminal after the script starts. Open this URL in the browser to access the Gradio user interface.

Model selection and loading:

Local model: Fill in the full path of your .nemo model file in the local model path input box, then click on the loading local model button.

Cloud End Model: Click the "Build Cloud End Model" button to download and load the default Parakeet model from NVIDIA NGC.

Models are displayed in text boxes below.

Adjust the length of the audio segment:
Use the slider to adjust the length of the audio segment in seconds. Larger segments can retain more context but may increase processing time and memory consumption. The recommended range is 60-180 seconds. This setting is saved with the model selection next time you click any Load Model button.

Upload files and generate subtitles:

From Video Generation: Switch to the "Subtitle From Video" tab page, click on the video upload area to upload your video file, and click on the "Start from Video Generation SRT" button.

From Audio Generation: Switch to the "Subtitles from Audio" tab, click on the audio upload area to upload your audio file, and click on the "Start SRT" button.

View and download results:

The processing status is updated in real time.

After processing, you can preview the subtitles generated in the SRT Subtitles area and click Download SRT Files to download the .srt subtitle files.

Attention

It may take longer to process large files or run on CPUs.

When you first load a cloud-end model, you need to download the model file, depending on your network speed.

If you have a ffmpeg-related error, make sure that FFmpeg is installed and configured correctly.

If the script hint is running on CPU, but you have NVIDIA GPU and want to use it, check whether PyTorch has been correctly installed as CUDA version (reference point 3 of "Step of installation " ).

