import numpy as np
import sounddevice as sd
import soundfile as sf

def select_device(input_only=True, use_pipewire=True):
        """Print all available audio devices with their information."""
        print("\nAvailable Audio Devices:")
        print("-" * 60)
        devices = sd.query_devices()
        valid_devices = []
        pipewire_device = None
        for i, device in enumerate(devices):
            if input_only and not device['max_input_channels']:
                continue
            if "pipewire" in device['name'].lower():
                pipewire_device = i
            valid_devices.append(device)

        if use_pipewire:
            if pipewire_device is not None:
                print(f"Using pipewire device: {devices[pipewire_device]['name']}")
                return pipewire_device
            else:
                print("No pipewire device found")
                return None

        elif valid_devices:
            for i, device in enumerate(valid_devices):
                print(f"Device {i}: {device['name']}")
                print(f"  Max Input Channels: {device['max_input_channels']}")
                print(f"  Max Output Channels: {device['max_output_channels']}")
                print(f"  Default Sample Rate: {device['default_samplerate']}")
                print("-" * 60)
        
            while True:
                try:
                    device_id = input("\nEnter device number to use for recording (or press Enter for default): ").strip()
                    if device_id == "":
                        print("Using default input device")
                        return None
                    try:
                        device_id = int(device_id)
                    except:
                        print('Only integers allowed, try again')
                        raise ValueError

                    try:
                        print(f"Using input device: {sd.query_devices(device_id)['name']}")
                        return device_id
                    except:
                        print('Invalid device number, try again')
                        raise KeyError
                except (KeyError, ValueError):
                    continue
                
        else:
            print("No valid input devices found")
            return None

def record(duration=5, samplerate=48000, device=None, playback=True, save_file=None, verbose=False):
    channels = 1
    
    vprint = lambda text: print(text) if verbose else None

    # Set input device if specified
    if device is not None:
        sd.default.device[0] = device
        vprint(f"Using input device: {sd.query_devices(device)['name']}")
    vprint(f"\nRecording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype=np.float32
    )
    sd.wait()

    vprint(f"Recording finished. Playing back audio...")
    sd.play(recording, samplerate)
    sd.wait()  # Wait for playback to finish

    if save_file is not None:
        sf.write(save_file, recording, samplerate)
        vprint(f"Recording saved to {save_file}")
    return recording

if __name__ == "__main__":
    record(device=select_device(), duration=5, verbose=True, save_file='test.wav')