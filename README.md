
# parakeet-tdt-0.6b-v2-SRT-GUI - A NeMo-based SRT Subtitle Generation Tool for Video/Audio

This project uses the nvidia/parakeet-tdt-0.6b-v2 ASR (Automatic Speech Recognition) model to automatically generate timestamped SRT subtitle files from video or audio files. The interface is built with Gradio, making it easy for users to upload files and get results.

## Main Features

  * Extract audio from various common video formats (e.g., MP4, MKV, AVI) and generate SRT subtitles.
  * Directly process various common audio formats (e.g., MP3, WAV, M4A, FLAC) to generate SRT subtitles.
  * Supports long video and long audio inputs.
  * Supports loading the pre-trained Parakeet cloud model from NVIDIA NGC (defaults to `nvidia/parakeet-tdt-0.6b-v2`).
  * Supports loading local `.nemo` model files from the user.
  * Adjustable audio processing chunk length to balance speed and contextual coherence.
  * Automatically detects a CUDA GPU and prioritizes its use for accelerated processing; runs on the CPU if no GPU is available (slower).
  * User-friendly interface with simple operations.
  * Automatically saves the user's selected model and chunk length configuration.

## System Requirements

  * Python 3.12.2 or higher, to ensure compatibility with the latest NeMo library.
  * **FFmpeg**: Used for audio/video decoding, encoding, and format conversion. **Must be installed separately and added to the system's PATH environment variable.**
  * NVIDIA GPU (recommended for acceleration, requires CUDA drivers). It can also run on a CPU if no GPU is present, but it will be very slow.

## Simplified Installation Steps (Windows)

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/NINIYOYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
    ```

2.  **Double-click to open install\_dependencies.bat**
    It will create and activate a Python virtual environment while also checking for and installing dependencies.
    The installation of torch depends on whether you need GPU acceleration.

3.  **Install FFmpeg:**
    This project relies on FFmpeg for audio extraction and preprocessing. You need to install it separately and ensure its executable path is added to the system's PATH environment variable.

      * **Windows:**
        1.  Download a pre-compiled version from the official FFmpeg download page ([https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)) (e.g., builds from "gyan.dev" or "BtbN").
        2.  Unzip the downloaded file.
        3.  Add the path to the `bin` directory inside the unzipped folder (e.g., `C:\ffmpeg\bin`) to your system's `Path` environment variable.
      * **Linux (Ubuntu/Debian, etc.):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
      * **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

4.  **Double-click to open launcher.bat**
    If your environment and dependencies are all installed correctly, you can run this project directly by double-clicking launcher.bat.

## Installation Steps

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/NINIYOYYO/parakeet-tdt-0.6b-v2-SRT-GUI.git
    ```

2.  **Create and activate a Python virtual environment (highly recommended):**

    ```bash
    python -m venv .venv
    ```

      * Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
      * macOS / Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install PyTorch (Important: GPU users, pay special attention\!):**
    If you want to use an NVIDIA GPU for accelerated processing (highly recommended), **be sure to manually install a version of PyTorch compatible with your CUDA environment before installing other dependencies.**

      * Press Win+R to open the Windows Run dialog, type CMD, and press Enter to open a terminal. In the terminal, type:

    <!-- end list -->

    ```bash
    nvidia-smi
    ```

    and press Enter to check your CUDA Version:

      * Visit the [official PyTorch installation guide page](https://pytorch.org/get-started/locally/).
      * Select the correct installation command based on your operating system, package manager (pip recommended), compute platform (e.g., CUDA 11.8, CUDA 12.1), and Python version.
      * For example, if you are using `pip` and your system has a CUDA 12.1 environment, you can run:
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    If you skip this step, or if your system does not have an NVIDIA GPU, the `nemo_toolkit` installed later may default to installing a CPU-only version of PyTorch.

4.  **Install project dependencies:**
    After activating the virtual environment and (optionally) installing a specific version of PyTorch, run:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Install FFmpeg:**
    This project relies on FFmpeg for audio extraction and preprocessing. You need to install it separately and ensure its executable path is added to the system's PATH environment variable.

      * **Windows:**
        1.  Download a pre-compiled version from the official FFmpeg download page ([https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)) (e.g., builds from "gyan.dev" or "BtbN").
        2.  Unzip the downloaded file.
        3.  Add the path to the `bin` directory inside the unzipped folder (e.g., `C:\ffmpeg\bin`) to your system's `Path` environment variable.
      * **Linux (Ubuntu/Debian, etc.):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
      * **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

6.  **Double-click to open launcher.bat**
    If your environment and dependencies are all installed correctly, you can run this project directly by double-clicking launcher.bat.

## Loading a Model Locally

**If you want to load the nvidia/parakeet-tdt-0.6b-v2 model locally, when you first launch the application, enter the local path of the model in the "Local Model Path (.nemo file)" input field.**
**For example: C:\\Users\\models--nvidia--parakeet-tdt-0.6b-v2\\snapshots\\30c5e6f557f6ba26e5819a9ed2e86f670186b43f\\parakeet-tdt-0.6b-v2.nemo**

## Interface Showcase

## How to Use

Ensure you have completed the environment setup and dependency installation as described in the steps above.

In the project's root directory, run the main script.

There are two ways to launch the application:

1.  **Run by launching launcher.bat**

2.  **Launch from the terminal (after entering the virtual environment)**

    ```bash
    python main.py
    ```

After the script starts, it will print a local URL in the terminal (usually http://127.0.0.1:7860 or a similar address). Open this URL in your browser to access the Gradio user interface.

### Model Selection and Loading:

  * **Local Model**: In the "Local Model Path" input box, enter the full path to your .nemo model file, then click the "Load Local Model" button.

  * **Cloud Model**: Simply click the "Load Cloud Model" button, and the default Parakeet model will be downloaded and loaded from NVIDIA NGC.

The model's loading status will be displayed in the text box below.

### Adjusting the Audio Chunk Length:

Use the slider to adjust the "Audio Chunk Length (seconds)". Larger chunks can preserve more context but may increase processing time and memory consumption. The recommended range is 60-180 seconds. This setting will be saved along with your model choice the next time you click either "Load Model" button.

### Uploading a File and Generating Subtitles:

  * **From Video**: Switch to the "Generate Subtitles from Video" tab, upload your video file by clicking on the video upload area, and then click the "Start Generating SRT from Video" button.

  * **From Audio**: Switch to the "Generate Subtitles from Audio" tab, upload your audio file by clicking on the audio upload area, and then click the "Start Generating SRT from Audio" button.

### Viewing and Downloading Results:

The processing status will update in real-time.

Once processing is complete, you can preview the generated subtitle content in the "SRT Subtitle Results" area and download the .srt subtitle file by clicking the "Download SRT File" link.

## Notes

  * Processing large files or running on a CPU may take a significant amount of time. Please be patient.

  * When loading the cloud model for the first time, the model files need to be downloaded, and the time required will depend on your network speed.

  * If you encounter any ffmpeg-related errors, please ensure that FFmpeg is installed and configured correctly.

  * If the script indicates it is running on the CPU, but you have an NVIDIA GPU and wish to use it, please double-check that PyTorch was installed correctly for the CUDA version (refer to point 3 of the "Installation Steps").
