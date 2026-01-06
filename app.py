"""
F5-TTS Vietnamese Text-to-Speech Application
Main entry point for the Gradio-based web interface.
"""

import spaces
import os
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm

# Import internal F5-TTS inference utilities
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)

# --- Configuration & Authentication ---

# Create organized samples directory structure
SAMPLES_DIR = "samples"
SAMPLES_WAVS_DIR = os.path.join(SAMPLES_DIR, "wavs")
SAMPLES_TXT_DIR = os.path.join(SAMPLES_DIR, "txt")

os.makedirs(SAMPLES_WAVS_DIR, exist_ok=True)
os.makedirs(SAMPLES_TXT_DIR, exist_ok=True)

def get_sample_list():
    """Scans both samples/ and samples/wavs directory for audio files."""
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac")
    samples = []
    
    # Scan root
    if os.path.exists(SAMPLES_DIR):
        for f in os.listdir(SAMPLES_DIR):
            if f.lower().endswith(audio_extensions):
                samples.append(f)
                
    # Scan wavs subfolder
    if os.path.exists(SAMPLES_WAVS_DIR):
        for f in os.listdir(SAMPLES_WAVS_DIR):
            if f.lower().endswith(audio_extensions) and f not in samples:
                samples.append(f)
                
    return ["None"] + sorted(samples)

def on_sample_change(sample_name):
    """Callback when a sample is selected from the dropdown."""
    if not sample_name or sample_name == "None":
        return None, ""
    
    # Try different possible paths for the audio file
    audio_paths = [
        os.path.join(SAMPLES_WAVS_DIR, sample_name),
        os.path.join(SAMPLES_DIR, sample_name)
    ]
    
    audio_path = None
    for path in audio_paths:
        if os.path.exists(path):
            audio_path = path
            break
            
    if not audio_path:
        return None, ""
    
    # Try different possible paths for the text file
    base_name = os.path.splitext(sample_name)[0]
    text_paths = [
        os.path.join(SAMPLES_TXT_DIR, base_name + ".txt"),
        os.path.join(SAMPLES_DIR, base_name + ".txt")
    ]
    
    sample_text = ""
    for text_path in text_paths:
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                sample_text = f.read().strip()
            break
            
    return audio_path, sample_text

def refresh_samples():
    """Updates the choices in the sample selector dropdown."""
    return gr.update(choices=get_sample_list())

# Retrieve Hugging Face token from environment variables
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Log in to Hugging Face if a token is provided
if hf_token:
    login(token=hf_token)

def post_process(text):
    """
    Cleans up the normalized text by removing redundant punctuation and spaces.
    
    Args:
        text (str): Raw normalized text.
    Returns:
        str: Cleaned text ready for synthesis.
    """
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    text = text.replace(" , , ", " , ")
    text = " " + text + " "
    text = text.replace(" ,, ", " , ")
    text = " " + text + " "
    text = text.replace('"', "")
    return " ".join(text.split())

# --- Model Loading ---

# Load the vocoder (Vocos is the default)
vocoder = load_vocoder()

# Load the F5-TTS DiT model
# We use cached_path to download and cache model files from Hugging Face
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
    vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
)

@spaces.GPU # Use ZeroGPU if available on Hugging Face Spaces
def infer_tts(ref_audio_orig: str, ref_text_input: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):
    """
    Main inference function for TTS generation.
    
    Args:
        ref_audio_orig (str): Path to the reference audio file.
        ref_text_input (str): Optional manual transcription of the reference audio.
        gen_text (str): The target Vietnamese text to synthesize.
        speed (float): Playback speed multiplier (default 1.0).
    """

    # Input validation
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Please enter text content with less than 1000 words.")
    
    try:
        # Step 1: Pre-process reference audio and text (handles automatic transcription if ref_text_input is empty)
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text_input)
        
        # Step 2: Normalize the target text using vinorm and run the core inference process
        # vinorm converts numbers and special characters to spoken Vietnamese
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            post_process(TTSnorm(gen_text)).lower(), 
            model, 
            vocoder, 
            speed=speed
        )
        
        # Step 3: Save the resulting spectrogram to a temporary file for display
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        # Return audio data and spectrogram path to Gradio
        return (final_sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# --- Premium UI Styling ---

CUSTOM_CSS = """
/* Background and Base Styles */
.gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
}

/* Glassmorphism Header */
.header-container {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    text-align: center;
}

.header-container h1 {
    background: linear-gradient(to right, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 10px;
}

/* Group/Card Styling */
.custom-card {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 15px !important;
    transition: all 0.3s ease;
}

.custom-card:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

/* Component Styling */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease, opacity 0.2s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    opacity: 0.9;
}

.gr-input, .gr-textbox, .gr-dropdown {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #f9fafb !important;
}

.gr-form {
    border: none !important;
    background: transparent !important;
}

/* Hide some default Gradio borders */
.gr-padded { padding: 0 !important; }
"""

# --- Gradio UI Layout ---

# Moving theme and css to launch() as recommended for Gradio 6.0+ compatibility
with gr.Blocks() as demo:
    with gr.Column(elem_classes="header-container"):
        gr.Markdown("""
        # 🎙️ F5-TTS Vietnamese: Next-Gen Speech Synthesis
        ### Trải nghiệm công nghệ AI tạo giọng nói tiếng Việt tự nhiên nhất dựa trên kiến trúc DiT tối tân.
        """)
    
    with gr.Tabs():
        with gr.TabItem("🚀 Synthesis"):
            with gr.Row():
                # LEFT COLUMN: Voice Source
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="custom-card"):
                        gr.Markdown("📂 **1. Nguồn Giọng Nói (Reference)**")
                        with gr.Row():
                            sample_selector = gr.Dropdown(
                                choices=get_sample_list(), 
                                label="Chọn từ bộ mẫu giọng", 
                                value="None",
                                scale=4
                            )
                            btn_refresh = gr.Button("🔄", scale=1)
                        
                        ref_audio = gr.Audio(
                            label="Tải lên hoặc ghi âm mẫu (10-15 giây)", 
                            type="filepath",
                            editable=True
                        )
                        
                        ref_text_input = gr.Textbox(
                            label="Văn bản của mẫu giọng (Tự động nếu để trống)", 
                            placeholder="Điền nội dung audio mẫu để tăng độ chính xác...", 
                            lines=3
                        )

                # RIGHT COLUMN: Target Text
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="custom-card"):
                        gr.Markdown("📝 **2. Nội Dung Cần Chuyển Đổi**")
                        gen_text = gr.Textbox(
                            label="Nhập văn bản tiếng Việt", 
                            placeholder="Nhập nội dung bạn muốn AI nói...", 
                            lines=8
                        )
                        
                        speed = gr.Slider(
                            minimum=0.3, maximum=2.0, value=1.0, step=0.1, 
                            label="⚡ Tốc độ nói (Speed)"
                        )
                        
                        btn_synthesize = gr.Button("🔥 Bắt Đầu Tạo Giọng Nói", variant="primary")

            # OUTPUT SECTION
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown("🎧 **3. Kết Quả (Generated Output)**")
                with gr.Row():
                    output_audio = gr.Audio(label="Bản âm thanh", type="numpy")
                    output_spectrogram = gr.Image(label="Âm phổ (Spectrogram)")

        with gr.TabItem("⚙️ Advanced Settings"):
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown("### 🛠️ Cấu Hình Nâng Cao (Sắp ra mắt)")
                gr.Checkbox(label="Sử dụng ODE Integration (Euler)", value=True, interactive=False)
                gr.Slider(label="NFE Steps", value=32, minimum=16, maximum=64, interactive=False)
                gr.Info("Lưu ý: Các thiết lập này hiện đang được tối ưu hóa tự động.")

        with gr.TabItem("📖 Guide & Docs"):
            with gr.Group(elem_classes="custom-card"):
                gr.Markdown(f"""
                ### ℹ️ Hướng dẫn & Lưu ý:
                - **Normalization**: Sử dụng `vinorm` để xử lý số, ngày tháng, ký tự đặc biệt theo cách đọc tiếng Việt.
                - **Voice Quality**: Nên sủ dụng mẫu giọng rõ ràng, không có tạp âm, độ dài từ 10-15 giây.
                - **Trình tự**: Chọn/Tải audio mẫu -> Nhập văn bản -> Nhấn nút 'Bắt Đầu'.
                
                ---
                👉 [Xem Tài Liệu Chi Tiết](file:///{os.path.abspath('DOCUMENTATION.md')})
                """)

    # UI Event Handlers
    btn_refresh.click(
        fn=refresh_samples,
        inputs=[],
        outputs=[sample_selector]
    )

    sample_selector.change(
        fn=on_sample_change,
        inputs=[sample_selector],
        outputs=[ref_audio, ref_text_input]
    )

    btn_synthesize.click(
        fn=infer_tts, 
        inputs=[ref_audio, ref_text_input, gen_text, speed], 
        outputs=[output_audio, output_spectrogram]
    )

# Run the Gradio application
# demo.queue() enables the request queue for handling multiple users
if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
