#Record 
from kokoro import KPipeline
from queue import Queue
import sounddevice as sd
import numpy as np
import whisper
import time
import threading

def record(stop, dataqueue): 
    """ Records audio data from the user's microphone and adds it into a queue for further processing.
    It takes as arguments:
    - stop: A event that (when set) signals to the function to stop recording
    - dataqueue: A queue to which the recorded audio data will be added.
    """ 
    def callback(indata, frames, time, status):
        if status:
            print(status)
        dataqueue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop.is_set():
            time.sleep(0.1)

stt = whisper.load_model("base")

def transcribe(audio_file: np.ndarray) -> str:
    """ Transcribes the given audio data using the Whispher speech recognition model.
    It takes as arguments:
    - audio_file : The audio to be transcribed
    Returns:
    - str: The transcribed text
    """
    extract = stt.transcribe(audio_file, fp16=False)
    text = extract["text"].strip()
    return text

kokoro_pipeline = KPipeline(lang_code='a')

tts_queue = Queue()

def tts():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        text = text.strip()
        if text:
            try:
                generator = kokoro_pipeline(text, voice='af_heart', speed=1.0)
                all_audio = []
                for m in generator:
                    all_audio.append(m[2])
                final_audio = np.concatenate(all_audio)
                sd.play(final_audio, samplerate=24000)
                sd.wait()
            except Exception as e:
                print(f"[tts] {e}")
        tts_queue.task_done()

threading.Thread(target=tts, daemon=True).start()

def texttospeech(text: str) -> None:
    if text.strip():
        tts_queue.put(text)