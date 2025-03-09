import torch
import torchaudio
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
from threading import Event, Thread, Lock
import threading
from queue import Queue
from dataclasses import dataclass
from enum import Enum, auto
import re
import nltk
nltk.download('punkt')

import vad
from models import build_model
from kokoro import generate
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from utils import select_device

class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = perf_counter() - self.start
        self.readout = f'Time: {self.elapsed:.3f} seconds'

# This and TTSbase kind of just add boilerplate for now but should be useful for adding more models
class STTBase:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rate = None

    def transcribe(self) -> Tuple[List[Tuple[str, float]], dict]:
        raise NotImplementedError
    
class WhisperSTT(STTBase):
    def __init__(self, model_name: str = "turbo", compute_type: str = "float16"):
        super().__init__()
        self.model = WhisperModel(model_size_or_path=model_name, device=self.device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> Tuple[List[Tuple[str, float]], dict]:
        return self.model.transcribe(audio)

class TTSBase:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rate = None

    def chunk_text(self, text: str) -> List[str]:
        return nltk.tokenize.sent_tokenize(text)

    def generate(self) -> np.ndarray:
        raise NotImplementedError

class KokoroTTS(TTSBase):
    def __init__(self):
        super().__init__()
        self.rate = 24_000
        # Build model
        self.model = build_model('models/kokoro-v0_19.pth', self.device)
        # Load voice
        self.VOICE_NAME = 'bf_emma' # Default voice (50-50 mix of Bella & Sarah)
        self.VOICEPACK = torch.load(f'voices/{self.VOICE_NAME}.pt', map_location=self.device)
        self.generate("Warmup") # Loads model into memory

    def generate(self, text: str, speed: float = 1.8, verbose: bool = False) -> np.ndarray:
        chunks = self.chunk_text(text)
        audio = np.array([])
        for chunk in chunks:
            with Timer() as t:
                chunk_audio, phonemes = generate(self.model, chunk, self.VOICEPACK, lang=self.VOICE_NAME[0], speed=speed)
            audio = np.concatenate((audio, chunk_audio))
            if verbose:
                print(f'Generated {len(chunk_audio) / self.rate:.2f} s of audio in {t.elapsed:.2f} s')
        return audio
    
class ZonosTTS(TTSBase):
    def __init__(self, condfile: str):
        super().__init__()
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
        self.rate = self.model.autoencoder.sampling_rate
        self.make_speaker(condfile=condfile)

    def make_speaker(self, condfile: str):
        wav, sampling_rate = torchaudio.load(condfile)
        self.speaker = self.model.make_speaker_embedding(wav, sampling_rate)

    def generate(self, text: str):
        chunks = self.chunk_text(text)
        audio = np.array([])
        for chunk in chunks:
            cond_dict = make_cond_dict(chunk, speaker=self.speaker, language="en-us")
            conditioning = self.model.prepare_conditioning(cond_dict)
            codes = self.model.generate(conditioning)
            chunk_audio = self.model.autoencoder.decode(codes).cpu().numpy().squeeze()
            audio = np.concatenate((audio, chunk_audio))
        return audio

class LlasaTTS(TTSBase):
    def __init__(self, 
                 llm_path: str = "HKUSTAudio/Llasa-3B",
                 codec_model_path: str = "HKUST-Audio/xcodec2",
                 condfile: str = "Anna.wav",
                 prompt_text: str = "A chance to leave him alone, but... No. She just wanted to see him again. Anna, you don't know how it feels to lose a sister. Anna, I'm sorry, but your father asked me not to tell you anything.",
                 device: str = None,
                 max_length: int = 2048,
                 temperature: float = 0.8,
                 top_p: float = 1.0):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from xcodec2.modeling_xcodec2 import XCodec2Model
        import torch
        import soundfile as sf
        import librosa
        import re

        # Set device if not provided
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.rate = 16000  # Llasa uses 16kHz sampling rate
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_text = prompt_text
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path).eval().to(self.device)
        self.codec = XCodec2Model.from_pretrained(codec_model_path).eval().to(self.device)
        
        # Load prompt data
        self.prompt_wav = self._load_audio(condfile)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and prepare audio file for the model"""
        import torch
        import soundfile as sf
        import librosa
        
        # Load audio file with original sample rate
        waveform, sr = sf.read(path)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            # Using librosa for resampling
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=16000)
        
        return torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)

    def _ids_to_speech_tokens(self, speech_ids: list) -> list:
        """Convert speech IDs to token format expected by the model"""
        return [f"<|s_{sid}|>" for sid in speech_ids]

    def _extract_speech_ids(self, tokens: list) -> list:
        """Extract speech IDs from token strings"""
        return [int(t[4:-2]) for t in tokens if t.startswith('<|s_') and t.endswith('|>')]

    def set_prompt(self, prompt_audio_path: str, prompt_text: str):
        """Update the conditioning prompt audio and text"""
        self.prompt_wav = self._load_audio(prompt_audio_path)
        self.prompt_text = prompt_text

    def generate(self, text: str, verbose: bool = False) -> np.ndarray:
        """Generate speech for the given text using the Llasa model"""
        import torch
        import time
        import numpy as np
        import re
        
        start_time = time.time()
        input_text = self.prompt_text + text

        with torch.no_grad():
            # Encode prompt audio
            vq_code = self.codec.encode_code(self.prompt_wav)[0, 0, :]
            speech_prefix = self._ids_to_speech_tokens(vq_code.tolist())

            # Format input text
            formatted_text = (
                f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            )

            # Create chat template
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": f"<|SPEECH_GENERATION_START|>{''.join(speech_prefix)}"}
            ]
            
            # Tokenize and generate
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt', continue_final_message=True
            ).to(self.device)
            
            outputs = self.llm.generate(
                input_ids,
                max_length=self.max_length,
                eos_token_id=self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>'),
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
            )

            # Process output
            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_ids = self._extract_speech_ids(speech_tokens)
            
            # Decode to audio
            speech_tensor = torch.tensor(speech_ids, device=self.device).unsqueeze(0).unsqueeze(0)
            waveform = self.codec.decode_code(speech_tensor)[0, 0, :].cpu().numpy()

        generation_time = time.time() - start_time

        if verbose:
            audio_length = len(waveform) / self.rate  # Calculate duration in seconds
            print(f"Generated speech in {generation_time:.2f} seconds")
            print(f"Audio length: {audio_length:.2f} seconds")

        return waveform

    def generate_batched(self, text: str, verbose: bool = False) -> np.ndarray:
        """Generate speech by splitting text into sentences for better quality"""
        import numpy as np
        import re
        
        combined_waveform = np.array([])
        total_time = 0.0
        
        # Simple sentence splitting logic
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty strings
                
            start_time = time.time()
            waveform = self.generate(
                text=sentence,
                verbose=verbose
            )
            total_time += time.time() - start_time
            
            # Concatenate waveforms
            combined_waveform = np.concatenate(
                (combined_waveform, waveform)
            ) if combined_waveform.size else waveform

        if verbose:
            audio_length = len(combined_waveform) / self.rate
            print(f"Total generation time: {total_time:.2f} seconds")
            print(f"Combined audio length: {audio_length:.2f} seconds")

        return combined_waveform


class SileroVAD:
    def __init__(self, send_audio_callback, voice_detected_callback=None, input_device=None, max_pause=1500, vad_thresh=0.5):
        self.send_audio_callback = send_audio_callback
        self.voice_detected_callback = voice_detected_callback
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
                     if self.voice_detected_callback:
                        self.voice_detected_callback()
                    
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

class LLM:
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

class VoiceAssistant:
    def __init__(self, tts: TTSBase, stt: STTBase, llm: LLM, input_device=None, allow_interrupt=True):
        self.allow_interrupt = allow_interrupt
        if input_device is not None:
            self.device_id = input_device
        else:
            self.device_id = select_device()
            
        # Simplified interrupt handling
        self.should_interrupt = False
        self.interrupt_lock = Lock()
        self.current_stream = None

        # Create VAD with callback that checks speaking state
        self.vad = SileroVAD(
            send_audio_callback=self.receive_audio,
            voice_detected_callback=self.handle_voice_detected,
            input_device=input_device,
            max_pause=300
        )
        self.tts: TTSBase = tts
        self.stt: STTBase = stt
        self.llm: LLM = llm
        self.stop_event = Event()
        self.processing_thread = None
        self.processing_cancelled = False
        self.pending_transcription = ""
        self.processing_lock = Lock()

        self.last_user_speech = time.time()

    def receive_audio(self, audio_list):
        if len(audio_list) < 2:
            return
        arr = np.array(np.concatenate(audio_list)).squeeze()
        segs, info = self.stt.transcribe(audio=arr)
        segs = list(segs)
        if segs:
            new_text = segs[0].text
            print(f'User: {new_text}')
            with self.processing_lock:
                if self.processing_thread is not None and self.processing_thread.is_alive():
                    self.pending_transcription += " " + new_text
                    self.processing_cancelled = True
                    return
                else:
                    self.pending_transcription = new_text
                    self.processing_cancelled = False
                    self.processing_thread = Thread(target=self.process_current_transcription)
                    self.processing_thread.start()

    def process_current_transcription(self):
        with self.processing_lock:
            current_text = self.pending_transcription
        # print(f'Processing: {current_text}')
        response = self.llm.process_message(current_text)
        with self.processing_lock:
            if self.processing_cancelled:
                # Remove the last user and assistant messages from the conversation memory
                if hasattr(self.llm.memory, 'chat_memory') and hasattr(self.llm.memory.chat_memory, 'messages'):
                    if len(self.llm.memory.chat_memory.messages) >= 2:
                        self.llm.memory.chat_memory.messages = self.llm.memory.chat_memory.messages[:-2]
                self.processing_cancelled = False
                self.processing_thread = Thread(target=self.process_current_transcription)
                self.processing_thread.start()
            else:
                print(f'Assistant: {response}')
                self.play_response(response)
                self.pending_transcription = ""
                self.processing_thread = None

    def play_audio_thread(self, audio):
        try:
            with self.interrupt_lock:
                self.should_interrupt = False

            # Use simple blocking play with interrupt checks
            self.current_stream = sd.OutputStream(
                samplerate=self.tts.rate,
                channels=1,
                blocksize=1024,
                device=self.device_id
            )
            
            print(f'elapsed: {(time.time() - self.last_user_speech) * 1000:.0f} ms')
            with self.current_stream:
                self.current_stream.start()
                sd.play(audio, samplerate=self.tts.rate)
                
                while self.current_stream.active:
                    # Check for interrupts every 50ms
                    sd.sleep(50)
                    with self.interrupt_lock:
                        if self.should_interrupt:
                            sd.stop()
                            break

                self.current_stream.stop()
                
        except Exception as e:
            print(f"Error in audio playback: {e}")
        finally:
            with self.interrupt_lock:
                self.should_interrupt = False
            self.current_stream = None

    def handle_voice_detected(self):
        self.last_user_speech = time.time()
        if not self.allow_interrupt:
            return
        with self.interrupt_lock:
            if self.current_stream is not None and self.current_stream.active:
                self.should_interrupt = True
        with self.processing_lock:
            if self.processing_thread is not None and self.processing_thread.is_alive():
                self.processing_cancelled = True

    def play_response(self, text):
        audio = self.tts.generate(text)
        playback_thread = Thread(target=self.play_audio_thread, args=(audio,))
        playback_thread.start()

    def active(self):
        print('Listening...')
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()