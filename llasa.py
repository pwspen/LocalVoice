from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import time
from typing import Optional
import sounddevice as sd
import re
from xcodec2.modeling_xcodec2 import XCodec2Model
import numpy as np
import librosa

class SpeechGenerator:
    def __init__(
        self,
        llm_path: str = "HKUSTAudio/Llasa-3B",
        codec_model_path: str = "HKUST-Audio/xcodec2",
        device: str = "cuda",
        input_device: int = None,
        prompt_audio_path: str = "Anna.wav",
        prompt_text: str = "A chance to leave him alone, but... No. She just wanted to see him again. Anna, you don't know how it feels to lose a sister. Anna, I'm sorry, but your father asked me not to tell you anything.",
        max_length: int = 2048,
        temperature: float = 0.8,
        top_p: float = 1.0
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_text = prompt_text
        self.input_device = input_device
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path).eval().to(device)
        self.codec = XCodec2Model.from_pretrained(codec_model_path).eval().to(device)
        
        # Load prompt data
        self.prompt_wav = self._load_audio(prompt_audio_path)
        self.input_text = None

    def _load_audio(self, path: str) -> torch.Tensor:
        # Load audio file with original sample rate
        waveform, sr = sf.read(path)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            # Using librosa for resampling
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=16000)
        
        return torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)

    def _ids_to_speech_tokens(self, speech_ids: list) -> list:
        return [f"<|s_{sid}|>" for sid in speech_ids]

    def _extract_speech_ids(self, tokens: list) -> list:
        return [int(t[4:-2]) for t in tokens if t.startswith('<|s_') and t.endswith('|>')]

    def generate_speech(self, text: str, auto_play: bool = True, 
                       output_path: Optional[str] = "gen.wav", 
                       verbose: bool = False) -> np.ndarray:
        """Generate speech and return waveform"""
        start_time = time.time()
        
        self.input_text = self.prompt_text + text

        with torch.no_grad():
            # Encode prompt audio
            vq_code = self.codec.encode_code(self.prompt_wav)[0,0,:]
            speech_prefix = self._ids_to_speech_tokens(vq_code.tolist())

            # Format input text
            formatted_text = (
                f"<|TEXT_UNDERSTANDING_START|>{self.input_text}<|TEXT_UNDERSTANDING_END|>"
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
            waveform = self.codec.decode_code(speech_tensor)[0,0,:].cpu().numpy()

        generation_time = time.time() - start_time

        # Save and play
        if output_path:
            sf.write(output_path, waveform, 16000)
        if auto_play:
            sd.play(waveform, samplerate=16000)
            sd.wait()

        if verbose:
            audio_length = len(waveform) / 16000  # Calculate duration in seconds
            print(f"Generated speech in {generation_time:.2f} seconds")
            print(f"Audio length: {audio_length:.2f} seconds")

        return waveform

    def set_prompt(self, prompt_audio_path: str, prompt_text: str):
        """Update the conditioning prompt audio and text"""
        self.prompt_wav = self._load_audio(prompt_audio_path)
        self.prompt_text = prompt_text

    def generate_speech_batched(self, text: str, auto_play: bool = True, 
                               output_path: Optional[str] = "gen_batched.wav", 
                               verbose: bool = False) -> np.ndarray:
        """Batch generate speech by splitting text into sentences"""
        combined_waveform = np.array([])
        total_time = 0.0
        
        # Simple sentence splitting logic
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty strings
                
            start_time = time.time()
            waveform = self.generate_speech(
                text=sentence,
                auto_play=False,
                output_path=None,
                verbose=verbose
            )
            total_time += time.time() - start_time
            
            # Concatenate waveforms
            combined_waveform = np.concatenate(
                (combined_waveform, waveform)
            ) if combined_waveform.size else waveform

        # Save and play combined audio
        if output_path:
            sf.write(output_path, combined_waveform, 16000)
        if auto_play:
            sd.play(combined_waveform, samplerate=16000)
            sd.wait()

        if verbose:
            audio_length = len(combined_waveform) / 16000
            print(f"Total generation time: {total_time:.2f} seconds")
            print(f"Combined audio length: {audio_length:.2f} seconds")

        return combined_waveform

    def record_new_prompt(self, speech_text: str, output_path: str = "new_prompt.wav", 
                         sample_rate: int = 48000, countdown: int = 3) -> None:
        """
        Record a new prompt speech and automatically apply it to the generator.
        """

        if self.input_device is None:
            self.select_device()

        print("\nPlease prepare to record the following text:")
        print(f"\n{speech_text}\n")
        
        input("Press Enter when you're ready to start the countdown...")
        
        # Countdown
        for i in range(countdown, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Speak now!")
        
        # Initialize recording
        recorded_audio = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            recorded_audio.append(indata.copy())
        
        # Set up recording stream
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, device=self.input_device):
                print("\nRecording... Press Enter to stop.")
                input()
        except Exception as e:
            print(f"Error during recording: {e}")
            return
        
        if not recorded_audio:
            print("No audio was recorded!")
            return
        
        # Convert recorded audio to numpy array
        recording = np.concatenate(recorded_audio, axis=0)
        
        # Normalize audio
        recording = recording / np.max(np.abs(recording))
        
        # Save the recording
        try:
            sf.write(output_path, recording, sample_rate)
            print(f"\nRecording saved to {output_path}")
            
            # Apply the new prompt
            self.set_prompt(output_path, speech_text)
            print("New prompt successfully applied!")
            
        except Exception as e:
            print(f"Error saving recording: {e}")

    @staticmethod
    def list_audio_devices():
        """Print all available audio devices with their information."""
        print("\nAvailable Audio Devices:")
        print("-" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}")
            print(f"  Max Input Channels: {device['max_input_channels']}")
            print(f"  Max Output Channels: {device['max_output_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']}")
            print("-" * 60)
        
        # print(f"\nDefault Input Device: {sd.query_devices(kind='input')}")
        # print(f"Default Output Device: {sd.query_devices(kind='output')}")
        return len(devices)
    
    def select_device(self):
        """Let user select an input device."""
        num_devices = self.list_audio_devices()
        
        while True:
            try:
                device_id = input("\nEnter device number to use for recording (or press Enter for default): ").strip()
                
                # if device_id == "":
                #     print("Using default input device")
                #     return None
                
                device_id = int(device_id)
                if 0 <= device_id < num_devices:
                    device = sd.query_devices(device_id)
                    if device['max_input_channels'] > 0:
                        print(f"Selected device: {device['name']}")
                        self.input_device = device_id
                        return True
                    else:
                        print("This device doesn't support input. Please select another.")
                else:
                    print("Invalid device number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

# Usage example
if __name__ == "__main__":

    skulls = """
    In our skulls we carry around 3 pounds of slimy, wet, greyish tissue, corrugated like crumpled toilet paper. You wouldn't think, to look at the unappetizing lump, that it was some of the most powerful stuff in the known universe.
    """

    combs = """Optical frequency combs were developed nearly two decades ago to support the world's most precise atomic clocks."""

    generator = SpeechGenerator(input_device=13,
                                prompt_audio_path="skulls.wav",
                                prompt_text=skulls)
    # Change above to None for posting in public - it auto configures then you can change it to your mic device id
    
    # generator.record_new_prompt(speech_text="Put the text you want to use here")
    
    # Example of updating prompt
    # generator.set_prompt(
    #     prompt_audio_path="Recording.mp3",
    #     prompt_text="This is my voice. This is Patrick speaking. Wow, what do I sound like? Wow. This is my voice."
    # )
    
    test_text = [
        "Five million years ago, the ancestors of lions ruled the day, the ancestors of wolves roamed the night.",
        "The ruling predators were armed with teeth and claws - sharp, hard cutting edges, backed up by powerful muscles.",
        "Humans (Homo sapiens) or modern humans are the most common and widespread species of primate, and the last surviving species of the genus Homo.",
        "Our model, Llasa, is a text-to-speech (TTS) system that extends the text-based LLaMA (1B,3B, and 8B) language model by incorporating speech tokens from the XCodec2 codebook, which contains 65,536 tokens.",
        "Optical frequency combs are specialized lasers that act like a ruler for light. They measure exact frequencies of light — from the invisible infrared and ultraviolet to visible red, yellow, green and blue light — quickly and accurately.",
        'def _ids_to_speech_tokens(self, speech_ids: list) -> list:\n  return [f"<|s_{sid}|>" for sid in speech_ids]',
        "Features appear to 'split' as we increase autoencoder size. When we gradually increase the width of the autoencoder from 512 (the number of neurons) to over 131,000 (256x), we find features which naturally fit together into families."
    ]
    for text in test_text:
        waveform = generator.generate_speech(text, verbose=True)