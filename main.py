import nemo.collections.asr as nemo_asr
import torch
import os
import subprocess
import time
from pydub import AudioSegment
import gradio as gr
import tempfile
import json

CONFIG_FILENAME = "config.json"
asr_model = None  # Initialize model variable
device = None     # Initialize device variable

# --- Configuration Management ---
def get_config_file_path():
    """Get the absolute path of the configuration file in the script directory."""
    try:
        # Find the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # If __file__ is not defined in an interactive environment
        script_dir = os.getcwd()
    return os.path.join(script_dir, CONFIG_FILENAME)

def save_config(local_model_path: str, chunk_length: int):
    """Save the current configuration to config.json."""
    config = {
        "local_model_path": local_model_path, # NGC is an empty string, local is a path, and None if never selected
        "chunk_length_s": chunk_length
    }
    config_file_path = get_config_file_path()
    try:
        with open(config_file_path, "w", encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4)
        print(f"Configuration saved to {config_file_path}")
    except Exception as e:
        print(f"Error: Failed to save configuration file '{config_file_path}': {e}")

def load_config() -> dict:
    """Load configuration from config.json.
    Returns a dictionary containing 'local_model_path' (can be None) and 'chunk_length_s'.
    """
    config_file_path = get_config_file_path()
    default_config = {"local_model_path": None, "chunk_length_s": 60} # None means no selection has been made yet

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding='utf-8') as config_file:
                loaded_config = json.load(config_file)
                # Ensure basic keys exist, providing default values if not
                loaded_config.setdefault("local_model_path", None)
                loaded_config.setdefault("chunk_length_s", 60)
                print(f"Configuration loaded from {config_file_path}: {loaded_config}")
                return loaded_config
        except json.JSONDecodeError:
            print(f"Error: Configuration file {config_file_path} is malformed. Using default configuration.")
            if os.path.exists(config_file_path): # Optional: back up the corrupted configuration file
                try:
                    os.rename(config_file_path, config_file_path + ".corrupted")
                    print(f"Backed up corrupted configuration file to {config_file_path}.corrupted")
                except Exception as e_mv:
                    print(f"Failed to back up corrupted configuration file: {e_mv}")
            return default_config
        except Exception as e:
            print(f"An unknown error occurred while loading configuration file '{config_file_path}': {e}. Using default configuration.")
            return default_config
    else:
        print(f"Configuration file {config_file_path} not found. Will use default settings (first run).")
        return default_config

# --- Model Loading ---
def load_asr_model_globally(
    local_model_path_to_try: str = None,
    load_from_ngc_explicitly: bool = False,
    save_choice_on_success: bool = False,
    current_chunk_value: int = 60
) -> str:
    global asr_model, device

    print("Attempting to load ASR model...")

    if torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("CUDA GPU detected, will run on GPU.")
    else:
        current_device = torch.device("cpu")
        print("Warning: No CUDA GPU detected, will run on CPU, which may be slow.")
    device = current_device
    asr_model = None # Reset model state before attempting to load

    # Scenario 1: Explicitly load from NGC
    if load_from_ngc_explicitly:
        print("Attempting to load model 'nvidia/parakeet-tdt-0.6b-v2' from NVIDIA NGC cloud...")
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2", map_location=device
            )
            status_msg = "Cloud model 'nvidia/parakeet-tdt-0.6b-v2' loaded successfully."
            print(status_msg)
            if save_choice_on_success:
                save_config(local_model_path="", chunk_length=current_chunk_value) # Use an empty path for NGC
            return status_msg
        except Exception as e:
            asr_model = None
            status_msg = f"Failed to load cloud model from NGC: {e}"
            print(status_msg)
            return status_msg

    # Scenario 2: Attempt to load from local_model_path_to_try
    if local_model_path_to_try and local_model_path_to_try.strip():
        actual_path = local_model_path_to_try.strip()
        if not os.path.exists(actual_path):
            status_msg = f"Error: The specified local model path does not exist: {actual_path}"
            print(status_msg)
            return status_msg # asr_model is already None
        if not actual_path.endswith(".nemo"):
            status_msg = f"Error: The specified local model path is not a valid .nemo file: {actual_path}"
            print(status_msg)
            return status_msg # asr_model is already None

        print(f"Attempting to load model from local path: {actual_path}...")
        try:
            asr_model = nemo_asr.models.ASRModel.restore_from(
                restore_path=actual_path, map_location=device
            )
            model_name = os.path.basename(actual_path)
            status_msg = f"Local model '{model_name}' loaded successfully."
            print(status_msg)
            if save_choice_on_success:
                save_config(local_model_path=actual_path, chunk_length=current_chunk_value)
            return status_msg
        except Exception as e:
            asr_model = None
            status_msg = f"Failed to load model from local path '{actual_path}': {e}"
            print(status_msg)
            return status_msg

    # Scenario 3: No valid or specific loading action taken
    status_msg = "Model not loaded. Please provide a valid local model path or choose to load from the cloud."
    # This message is more general if neither explicit NGC loading nor local path loading was handled.
    # If local_model_path_to_try was provided but resulted in an error, a specific error has already been returned above.
    if not load_from_ngc_explicitly and not (local_model_path_to_try and local_model_path_to_try.strip()):
         print(status_msg) # Log this specific case where no action was taken

    return status_msg


# --- Helper Functions (ffmpeg, audio processing, SRT generation) ---
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not detected or not installed correctly. Please install ffmpeg and ensure it is in the system's PATH.")
        return False

def extract_audio_from_video(input_video_path: str) -> str:
    if not check_ffmpeg(): return None
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_audio_path = temp_audio_file.name
    temp_audio_file.close()

    ffmpeg_command = [
        'ffmpeg', '-i', input_video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y', output_audio_path
    ]
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, errors='ignore')
        print(f"Audio extracted successfully to: {output_audio_path}")
        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
            return output_audio_path
        else:
            print(f"Error: ffmpeg command executed successfully, but no audio file was generated: {output_audio_path}")
            if os.path.exists(output_audio_path): os.remove(output_audio_path) # Clean up the empty file
            return None
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error during audio extraction: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        if os.path.exists(output_audio_path): os.remove(output_audio_path)
        return None
    except FileNotFoundError:
        print("Error: FFmpeg executable not found.")
        return None


def preprocess_direct_audio(input_audio_path: str) -> str:
    if not check_ffmpeg(): return None
    if not input_audio_path or not os.path.exists(input_audio_path):
        print(f"Error: The provided audio file path is invalid or the file does not exist: {input_audio_path}")
        return None

    temp_processed_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_wav_path = temp_processed_audio_file.name
    temp_processed_audio_file.close()

    ffmpeg_command = [
        'ffmpeg', '-i', input_audio_path, '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y', output_wav_path
    ]
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, errors='ignore')
        print(f"Direct audio preprocessing successful: {input_audio_path} -> {output_wav_path}")
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            return output_wav_path
        else:
            print(f"Error: ffmpeg command executed, but no valid audio file was generated: {output_wav_path}")
            if os.path.exists(output_wav_path): os.remove(output_wav_path) # Clean up
            return None
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error while preprocessing direct audio '{input_audio_path}': {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        if os.path.exists(output_wav_path): os.remove(output_wav_path)
        return None
    return None


def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio_in_chunks(model, audio_path: str, chunk_length_ms: int) -> list:
    if model is None:
        print("Model is not loaded, transcription cannot proceed.")
        return []
    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: Audio file path '{audio_path}' is invalid or the file does not exist.")
        return []

    print(f"Loading audio file '{audio_path}' for chunk processing...")
    try:
        audio = AudioSegment.from_wav(audio_path) # Assuming it has been preprocessed to WAV
        audio = audio.set_frame_rate(16000).set_channels(1) # Ensure format
    except Exception as e:
        print(f"Error loading or processing audio file '{audio_path}' (pydub): {e}")
        return []

    audio_duration_ms = len(audio)
    print(f"Total audio duration: {audio_duration_ms / 1000:.2f} seconds")
    all_segment_timestamps = []

    for i in range(0, audio_duration_ms, chunk_length_ms):
        start_time_ms = i
        end_time_ms = min(i + chunk_length_ms, audio_duration_ms)
        chunk = audio[start_time_ms:end_time_ms]
        
        temp_chunk_file_path = "" # Defined outside the try block
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk_file:
                temp_chunk_file_path = temp_chunk_file.name
            chunk.export(temp_chunk_file_path, format="wav")
            print(f"Processing audio chunk: {start_time_ms / 1000:.2f}s - {end_time_ms / 1000:.2f}s")

            chunk_output_list = model.transcribe([temp_chunk_file_path], batch_size=1, timestamps=True)

            if chunk_output_list and hasattr(chunk_output_list[0], 'timestamp') and \
               chunk_output_list[0].timestamp and 'segment' in chunk_output_list[0].timestamp:
                current_chunk_segments = chunk_output_list[0].timestamp['segment']
                chunk_global_start_offset_sec = start_time_ms / 1000.0
                for segment_data in current_chunk_segments:
                    local_start_sec = segment_data['start']
                    local_end_sec = segment_data['end']
                    text_content = segment_data.get('segment', segment_data.get('text', ''))
                    global_start_sec = local_start_sec + chunk_global_start_offset_sec
                    global_end_sec = local_end_sec + chunk_global_start_offset_sec
                    if global_end_sec < global_start_sec: # Safety check
                        global_end_sec = global_start_sec + 0.05
                    all_segment_timestamps.append({
                        'start': global_start_sec, 'end': global_end_sec, 'segment': text_content
                    })
            else:
                full_text = chunk_output_list[0].text if chunk_output_list else "N/A"
                print(f"Warning: Audio chunk failed to generate segment timestamps. Full transcription: '{full_text}'.")
        except Exception as e:
            print(f"Error transcribing audio chunk '{temp_chunk_file_path}': {e}")
            import traceback
            traceback.print_exc()
        finally:
            if temp_chunk_file_path and os.path.exists(temp_chunk_file_path):
                try:
                    os.remove(temp_chunk_file_path)
                except OSError as e_os:
                    print(f"OS error while deleting temporary audio file '{temp_chunk_file_path}': {e_os}")
    
    all_segment_timestamps.sort(key=lambda x: x['start'])
    return all_segment_timestamps

def generate_srt_content(segment_timestamps: list) -> str:
    srt_content = ""
    for i, stamp in enumerate(segment_timestamps):
        subtitle_number = i + 1
        start_time_srt = format_srt_time(stamp['start'])
        end_time_srt = format_srt_time(stamp['end'])
        segment_text = stamp['segment']
        srt_block = f"{subtitle_number}\n{start_time_srt} --> {end_time_srt}\n{segment_text}\n\n"
        srt_content += srt_block
    return srt_content

# --- Gradio Processing Functions ---
def process_video_for_srt(video_file_obj, chunk_length_s: int):
    if asr_model is None:
        yield "Error: ASR model is not loaded. Please load a model first.", None, ""
        return
    if video_file_obj is None:
        yield "Please upload a video file.", None, ""
        return
    
    input_video_path = video_file_obj # Gradio Video object has a .name attribute for the path
    print(f"Starting to process video file: {input_video_path}")
    start_time_total = time.time()
    extracted_audio_path = None
    output_srt_path_for_download = None # For the Gradio File component

    try:
        yield "Status: Extracting audio...", None, ""
        extracted_audio_path = extract_audio_from_video(input_video_path)
        if not extracted_audio_path:
            yield "Error: Audio extraction failed. Please check the video file or ffmpeg installation.", None, ""
            return

        chunk_length_ms = chunk_length_s * 1000
        yield f"Status: Transcribing audio (chunk size: {chunk_length_s} seconds)...", None, ""
        segment_timestamps = transcribe_audio_in_chunks(asr_model, extracted_audio_path, chunk_length_ms)

        if not segment_timestamps:
            yield "Error: Transcription did not generate valid segment timestamps.", None, ""
            return

        yield "Status: Generating SRT content...", None, ""
        srt_content = generate_srt_content(segment_timestamps)

        with tempfile.NamedTemporaryFile(mode="w", encoding='utf-8', suffix=".srt", delete=False) as tmp_srt_file:
            tmp_srt_file.write(srt_content)
            output_srt_path_for_download = tmp_srt_file.name
        
        elapsed_time_total = time.time() - start_time_total
        status_message = f"Processing complete. Total time taken: {elapsed_time_total:.2f} seconds."
        print(f"{status_message} SRT file is at: {output_srt_path_for_download}")
        yield status_message, output_srt_path_for_download, srt_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"An unknown error occurred during processing: {e}", None, ""
    finally:
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            try: os.remove(extracted_audio_path)
            except OSError as e_clean: print(f"Error cleaning up temporary audio file {extracted_audio_path}: {e_clean}")
        # Gradio will handle the deletion of output_srt_path_for_download (the temporary file it provides)

def process_audio_for_srt(audio_file_path_str, chunk_length_s: int): # Audio input type: "filepath"
    if asr_model is None:
        yield "Error: ASR model is not loaded. Please load a model first.", None, ""
        return
    if not audio_file_path_str:
        yield "Please upload an audio file.", None, ""
        return

    print(f"Starting to process audio file: {audio_file_path_str}")
    start_time_total = time.time()
    processed_audio_path = None
    output_srt_path_for_download = None

    try:
        if not check_ffmpeg(): # Check again, although the initial check is good
            yield "Error: FFMPEG is not installed or not in PATH, cannot process audio.", None, ""
            return

        yield "Status: Preprocessing audio file...", None, ""
        processed_audio_path = preprocess_direct_audio(audio_file_path_str)
        if not processed_audio_path:
            yield "Error: Audio preprocessing failed.", None, ""
            return
        
        chunk_length_ms = chunk_length_s * 1000
        yield f"Status: Preparing to transcribe audio (chunk size: {chunk_length_s} seconds)...", None, ""
        segment_timestamps = transcribe_audio_in_chunks(asr_model, processed_audio_path, chunk_length_ms)

        if not segment_timestamps:
            yield "Error: Transcription did not generate valid segment timestamps.", None, ""
            return

        yield "Status: Generating SRT content...", None, ""
        srt_content = generate_srt_content(segment_timestamps)

        with tempfile.NamedTemporaryFile(mode="w", encoding='utf-8', suffix=".srt", delete=False) as tmp_srt_file:
            tmp_srt_file.write(srt_content)
            output_srt_path_for_download = tmp_srt_file.name
            
        elapsed_time_total = time.time() - start_time_total
        status_message = f"Audio processing complete. Total time taken: {elapsed_time_total:.2f} seconds."
        print(f"{status_message} SRT file is at: {output_srt_path_for_download}")
        yield status_message, output_srt_path_for_download, srt_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"An unknown error occurred during audio processing: {e}", None, ""
    finally:
        if processed_audio_path and os.path.exists(processed_audio_path) and processed_audio_path != audio_file_path_str:
            try: os.remove(processed_audio_path)
            except OSError as e_clean: print(f"Error cleaning up preprocessed audio file {processed_audio_path}: {e_clean}")


# --- Gradio Interface Setup ---
if __name__ == "__main__":
    if not check_ffmpeg():
        print("Important: FFMPEG not found. Video and audio processing functions will be limited or will not work.")

    # --- Model Loading Logic on Startup ---
    config = load_config()
    # saved_model_path can be: a file path (str), "" (for NGC), or None (not previously selected)
    saved_model_path = config.get("local_model_path")
    initial_chunk_length = config.get("chunk_length_s", 60)
    initial_model_status = "Model not loaded. Please select a local model or load from the cloud." # Default value for a true first run

    if saved_model_path is not None: # A selection has been made previously
        if saved_model_path == "": # Last selection was NGC
            print("Last selection recorded in config was cloud NGC model, attempting to auto-reload...")
            initial_model_status = load_asr_model_globally(
                load_from_ngc_explicitly=True,
                save_choice_on_success=False, # Do not re-save the configuration on auto-load
                current_chunk_value=initial_chunk_length
            )
        elif os.path.exists(saved_model_path): # A local path was saved and it exists
            print(f"Local model path recorded in config: {saved_model_path}, attempting to auto-reload...")
            initial_model_status = load_asr_model_globally(
                local_model_path_to_try=saved_model_path,
                save_choice_on_success=False, # Do not re-save the configuration on auto-load
                current_chunk_value=initial_chunk_length
            )
        else: # A local path was saved, but it no longer exists
            error_msg = f"Error: The local model path '{saved_model_path}' in the config file was not found or is invalid. Model not loaded."
            print(error_msg)
            initial_model_status = error_msg

    else:
        # This is a true first run (config file does not exist or local_model_path is explicitly None)
        print("First run or no model previously configured. The model will not load automatically, please select one manually.")
        # initial_model_status remains the default "Model not loaded..."

    # --- Gradio UI Definition ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Automatic SRT Subtitle Generation Tool for Video/Audio (NeMo Parakeet)")
        gr.Markdown("Upload a video or audio file to automatically generate SRT subtitles. Based on the nvidia/parakeet-tdt-0.6b-v2 model.")

        with gr.Row():
            with gr.Column(scale=3):
                # If saved_model_path is None (first run) or "" (NGC), initialize the textbox with ""
                local_model_path_input = gr.Textbox(
                    label="Local Model Path (.nemo file)",
                    placeholder="e.g., /path/to/your_model.nemo",
                    value=saved_model_path if saved_model_path is not None else ""
                )
            with gr.Column(scale=1, min_width=150):
                load_local_model_button = gr.Button("Load Local Model", variant="secondary")
            with gr.Column(scale=1, min_width=150):
                load_cloud_model_button = gr.Button("Load Cloud Model", variant="primary")
        
        model_status_output = gr.Textbox(
            label="Model Load Status",
            value=initial_model_status,
            lines=2,
            interactive=False,
            max_lines=3
        )

        chunk_slider = gr.Slider(
            minimum=10, maximum=300, value=initial_chunk_length, step=5,
            label="Audio Chunk Length (seconds)",
            info="Recommended: 60-180 seconds. After changing, this setting will be saved along with the model choice the next time you click either 'Load Model' button."
        )
        gr.Markdown("---")

        with gr.Tab("Generate Subtitles from Video"):
            video_input = gr.Video(label="Upload Video File (e.g., MP4, MKV)")
            video_submit_button = gr.Button("Start Generating SRT from Video", variant="primary")
        
        with gr.Tab("Generate Subtitles from Audio"):
            # For audio, type="filepath" is generally more robust for ffmpeg processing
            audio_input = gr.Audio(label="Upload Audio File (e.g., MP3, WAV, M4A)", type="filepath")
            audio_submit_button = gr.Button("Start Generating SRT from Audio", variant="primary")

        status_output = gr.Textbox(label="Processing Status", lines=1, interactive=False)
        with gr.Accordion("SRT Subtitle Results", open=True):
            srt_file_output = gr.File(label="Download SRT File (.srt)", interactive=False)
            srt_preview_output = gr.Textbox(label="SRT Content Preview", lines=10, max_lines=20, interactive=False)

        # --- Button Click Handlers ---
        def handle_load_local_click(path_from_input_box, chunk_val_from_slider):
            if not path_from_input_box or not path_from_input_box.strip():
                return "Error: Please enter a valid local model path before clicking 'Load Local Model'. To load the cloud model, use the corresponding button."
            return load_asr_model_globally(
                local_model_path_to_try=path_from_input_box,
                load_from_ngc_explicitly=False,
                save_choice_on_success=True,
                current_chunk_value=chunk_val_from_slider
            )

        def handle_load_cloud_click(chunk_val_from_slider):
            return load_asr_model_globally(
                local_model_path_to_try=None,
                load_from_ngc_explicitly=True,
                save_choice_on_success=True,
                current_chunk_value=chunk_val_from_slider
            )

        load_local_model_button.click(
            fn=handle_load_local_click,
            inputs=[local_model_path_input, chunk_slider],
            outputs=[model_status_output]
        )
        load_cloud_model_button.click(
            fn=handle_load_cloud_click,
            inputs=[chunk_slider], # Only chunk_slider is needed to save the configuration
            outputs=[model_status_output]
        )

        video_submit_button.click(
            fn=process_video_for_srt,
            inputs=[video_input, chunk_slider],
            outputs=[status_output, srt_file_output, srt_preview_output]
        )
        audio_submit_button.click(
            fn=process_audio_for_srt,
            inputs=[audio_input, chunk_slider],
            outputs=[status_output, srt_file_output, srt_preview_output]
        )
        
        gr.Markdown("---")
        gr.Markdown("Note: Processing speed depends on your hardware (GPU/CPU) and the file size.")
        if device and device.type == 'cpu': # Check if the device has been initialized
            gr.Markdown("️️️⚠️ **Warning: Currently running on CPU, which will be very slow. It is recommended to use a CUDA GPU for better performance.**")
        elif not device and torch.cuda.is_available(): # Model not yet loaded, but GPU is available
             gr.Markdown("️️️ℹ️ **Info: CUDA GPU detected. The application will attempt to run on the GPU after a model is selected.**")
        elif not device: # Model not loaded, and no GPU
             gr.Markdown("️️️⚠️ **Warning: No CUDA GPU detected. The application will run on the CPU after a model is selected, which may be slow.**")

    print("Gradio interface is about to launch...")
    demo.launch()
    print("Gradio interface has stopped.")
