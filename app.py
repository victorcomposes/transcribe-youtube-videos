import whisper
import yt_dlp as youtube_dl
import gradio as gr
import os
import re
import logging

logging.basicConfig(level=logging.INFO)
model = whisper.load_model("small")

def get_text(url):
    #try:
    if url != '':
        output_text_transcribe = ''

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        out_file = info_dict['title'] + ' [' + info_dict['id'] + ']' + ".mp3"

    file_stats = os.stat(out_file)
    logging.info(f'Size of audio file in Bytes: {file_stats.st_size}')
    
    if file_stats.st_size <= 9999999999:
        result = model.transcribe(out_file, fp16=False)
        return result['text'].strip()
    else:
        logging.error('Videos for transcription on this space are limited to about 1.5 hours.')
    #finally:
    #    raise gr.Error("Exception: There was a problem transcribing the audio.")

def get_summary(article):
    first_sentences = ' '.join(re.split(r'(?<=[.:;])\s', article)[:5])
    b = summarizer(first_sentences, min_length = 20, max_length = 120, do_sample = False)
    b = b[0]['summary_text'].replace(' .', '.').strip()
    return b

with gr.Blocks() as demo:
    gr.Markdown("<h1><center>Free Fast YouTube URL Video-to-Text using <a href=https://openai.com/blog/whisper/ target=_blank>OpenAI's Whisper</a> Model</center></h1>")
    #gr.Markdown("<center>Enter the link of any YouTube video to generate a text transcript of the video and then create a summary of the video transcript.</center>")
    gr.Markdown("<center>Enter the link of any YouTube video to generate a text transcript of the video.</center>")
    gr.Markdown("<center><b>'Whisper is a neural net that approaches human level robustness and accuracy on English speech recognition.'</b></center>")
    gr.Markdown("<center>Transcription takes 5-10 seconds per minute of the video (bad audio/hard accents slow it down a bit). #patience<br />If you have time while waiting, drop a ♥️ and check out my <a href=https://www.artificial-intelligence.blog target=_blank>AI blog</a> (opens in new tab).</center>")
    
    input_text_url = gr.Textbox(placeholder='Youtube video URL', label='YouTube URL')
    result_button_transcribe = gr.Button('Transcribe')
    output_text_transcribe = gr.Textbox(placeholder='Transcript of the YouTube video.', label='Transcript')
    
    #result_button_summary = gr.Button('2. Create Summary')
    #output_text_summary = gr.Textbox(placeholder='Summary of the YouTube video transcript.', label='Summary')
    
    result_button_transcribe.click(get_text, inputs = input_text_url, outputs = output_text_transcribe)
    #result_button_summary.click(get_summary, inputs = output_text_transcribe, outputs = output_text_summary)

demo.queue(default_enabled = True).launch(debug = True)