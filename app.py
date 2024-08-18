import whisper
import yt_dlp as youtube_dl
import gradio as gr
import os
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Load Whisper model
model = whisper.load_model("base")

def download_audio(url):
    """Download audio from YouTube and return the file path."""
    if not url:
        raise ValueError("The URL cannot be empty.")
    
    # Define the output template for the file name
    outtmpl = '%(id)s.%(ext)s'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': outtmpl  # Set the output file name template
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        out_file = f"{info_dict['id']}.mp3"  # Use the id as the file name
    
    return out_file

def get_audio_file_size(file_path):
    """Return the size of the audio file in bytes."""
    return os.stat(file_path).st_size

def format_timestamp(seconds):
    """Convert seconds to YouTube-style timestamp (minutes:seconds)."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def transcribe_audio(file_path):
    """Transcribe audio file using Whisper model and include timestamps."""
    # Perform transcription
    result = model.transcribe(file_path, fp16=False)
    
    # Extract segments if they are available
    segments = result.get('segments', [])
    
    # Format the transcription with YouTube-style timestamps
    transcript_with_timestamps = []
    for segment in segments:
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        transcript_with_timestamps.append(f"[{start_time} - {end_time}] {text}")
    
    # If no segments are found, use the whole text
    if not segments:
        transcript_with_timestamps.append(result['text'])
    
    return "\n".join(transcript_with_timestamps)

def get_text(url):
    """Process the URL to get transcription text."""
    try:
        audio_file = download_audio(url)
        file_size = get_audio_file_size(audio_file)
        logging.info(f'Size of audio file in Bytes: {file_size}')
        
        if file_size <= 9999999999:  # Size check (limit for large files)
            return transcribe_audio(audio_file)
        else:
            logging.error('Videos for transcription are limited to about 1.5 hours.')
            return "Video exceeds size limit for transcription."

    except Exception as e:
        logging.error(f'Error occurred: {e}')
        return "An error occurred during transcription."

def create_gradio_ui():
    """Create and launch Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("<center>Enter the link of any YouTube video to generate a text transcript of the video.</center>")
        gr.Markdown("<center><b>'Whisper is a neural net that approaches human level robustness and accuracy on English speech recognition.'</b></center>")
        gr.Markdown("<center>Transcription takes 5-10 seconds per minute of the video (bad audio/hard accents slow it down a bit). #patience<br />If you have time while waiting, drop a ♥️ and check out my <a href=https://www.artificial-intelligence.blog target=_blank>AI blog</a> (opens in new tab).</center>")
        
        input_text_url = gr.Textbox(placeholder='YouTube video URL', label='YouTube URL')
        result_button_transcribe = gr.Button('Transcribe')
        output_text_transcribe = gr.Textbox(placeholder='Transcript of the YouTube video.', label='Transcript', lines=20)

        result_button_transcribe.click(get_text, inputs=input_text_url, outputs=output_text_transcribe)
    
    demo.queue(default_enabled=True).launch(debug=True)

if __name__ == "__main__":
    create_gradio_ui()
