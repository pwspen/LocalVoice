import torch
import sounddevice as sd
from sounddevice import CallbackFlags
import numpy as np
import time
from pathlib import Path
from typing import Any, Tuple, List
import queue
from time import perf_counter
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import Optional, Union
from faster_whisper import WhisperModel
import os

import vad
from models import build_model
from kokoro import generate

class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = perf_counter() - self.start
        self.readout = f'Time: {self.elapsed:.3f} seconds'

class Kokoro:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rate = 24_000
        print(f"Using device: {self.device}")
        # Build model
        self.MODEL = build_model('models/kokoro-v0_19.pth', self.device)
        # Load voice
        self.VOICE_NAME = 'af' # Default voice (50-50 mix of Bella & Sarah)
        self.VOICEPACK = torch.load(f'voices/{self.VOICE_NAME}.pt', map_location=self.device)
        self.generate("Warmup") # Loads model into memory

    def generate(self, text: str, speed: float = 1.8, verbose: bool = False):
        # Always returns 24kHz
        with Timer() as t:
            audio, phonemes = generate(self.MODEL, text, self.VOICEPACK, lang=self.VOICE_NAME[0], speed=speed)
        duration = len(audio) / self.rate
        if verbose:
            print(f'Generated {duration:.2f} s of audio in {t.elapsed:.2f} s')
        return audio

class SileroVAD:
    def __init__(self, send_audio_callback, input_device=None, max_pause=1500, vad_thresh=0.5):
        self.send_audio_callback = send_audio_callback
        self.path = "silero_vad.onnx"
        self.model = vad.VAD(model_path=str(Path.cwd() / "models" / self.path))
        self.VAD_SIZE = 50  # Milliseconds of sample for Voice Activity Detection (VAD)
        self.VAD_THRESHOLD = vad_thresh  # Threshold for VAD detection
        self.SAMPLE_RATE = 16000
        self.MAX_PAUSE = max_pause # Milliseconds of pause allowed in continuous speech
        self.MAX_NO_CONF = self.MAX_PAUSE / self.VAD_SIZE
        self.CURR_NO_CONF = self.MAX_NO_CONF
        self.KEEP_LAST = 10 # Samples of audio (of VAD size) to keep in rolling buffer
        self.recording = False
        self.samples: List[np.ndarray] = []

        self.input_stream = sd.InputStream(
                device=input_device,
                samplerate=self.SAMPLE_RATE,
                channels=2,  # Use stereo channels for loopback devices
                callback=self.audio_callback_for_sdInputStream,
                blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
            )
        self.input_stream.start()

    def audio_callback_for_sdInputStream(
                self, indata: np.ndarray, frames: int, time: Any, status: CallbackFlags
            ):
                # Convert to mono by averaging channels
                data = indata.copy().mean(axis=1)
                vad_value = self.model.process_chunk(data)
                vad_confidence = vad_value > self.VAD_THRESHOLD
    
                self.samples.append(data)

                if not self.recording and len(self.samples) > self.KEEP_LAST:
                    self.samples.pop(0)
                
                if vad_confidence:
                     self.CURR_NO_CONF = 0
                     if not self.recording:
                          self.recording = True
                elif self.CURR_NO_CONF < self.MAX_NO_CONF:
                    self.CURR_NO_CONF += 1
                    if self.CURR_NO_CONF == self.MAX_NO_CONF:
                         self.recording = False
                         self.send_audio_callback(self.samples)
                         self.samples = []          

    def __del__(self):
         self.input_stream.stop()

class LLMProcessor:
    def __init__(self, 
                 provider: str = "openrouter",
                 model: str = "mistralai/mistral-7b-instruct",
                 api_key: Optional[str] = None,
                 ollama_base_url: str = "http://localhost:11434",
                 system_prompt: str = "You are a helpful AI assistant. Keep your responses concise and natural. Note that the user's input is coming from speech to text so it may be a bit garbled, don't comment on it and do your best to interpret the intended speech."):
        
        self.provider = provider
        self.system_prompt = system_prompt
        
        # Initialize the LLM based on provider
        if provider == "openrouter":
            if not api_key:
                raise ValueError("API key required for OpenRouter")
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=model,
                streaming=True,
                temperature=0.7
            )
        elif provider == "ollama":
            self.llm = ChatOllama(
                base_url=ollama_base_url,
                model=model,
                streaming=True,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Setup conversation memory and prompt
        self.memory = ConversationBufferMemory()
        prompt_template = f"""
        {system_prompt}

        Current conversation:
        {{history}}
        Human: {{input}}
        Assistant:"""
        
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_template
        )
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
    
    def process_message(self, message: str) -> str:
        """Process a message through the LLM and return the response"""
        response = self.conversation.predict(input=message)
        return response
    
    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()

if __name__ == "__main__":
    print("Available input devices:")
    input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
    
    # Find PipeWire device
    pipewire_device = None
    for i, dev in enumerate(input_devices):
        print(f"{i}: {dev['name']}")
        if "pipewire" in dev['name'].lower():
            pipewire_device = i
            
    # If PipeWire is found, use it; otherwise prompt for selection
    if pipewire_device is not None:
        print(f"Automatically selected PipeWire device: {input_devices[pipewire_device]['name']}")
        device_id = input_devices[pipewire_device]['index']
    else:
        devnum = int(input("PipeWire not found. Select INPUT device number: "))
        device_id = input_devices[devnum]['index']

    tts = Kokoro()
    stt = WhisperModel("turbo", device="cuda", compute_type="float16")
    llm = LLMProcessor(
        provider="ollama",
        model="llama3.1:8b",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    def call(audio_list):
        print('Transcribing...')
        if len(audio_list) < 2:
            return
        arr = np.array(np.concatenate(audio_list)).squeeze()
        segs, info = stt.transcribe(audio=arr)
        segs = list(segs)
        # print(segs)
        if segs:
            transcribed_text = segs[0].text
            print(f'User: {transcribed_text}')
        #     assistant_text = llm.process_message(transcribed_text)
        #     print(f'Assistant: {assistant_text}')
        #     audio = tts.generate(assistant_text)
        #     sd.play(audio, samplerate=tts.rate)
        #     sd.wait()

    # Initialize VAD with loopback device
    vad = SileroVAD(send_audio_callback=call, input_device=device_id, max_pause=300)
    print("*"*10, 'Ready to process system audio', "*"*10,)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass