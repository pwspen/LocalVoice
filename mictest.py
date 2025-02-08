import sounddevice as sd
import numpy as np
import time

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

def select_device():
    """Let user select an input device."""
    num_devices = list_audio_devices()
    
    while True:
        try:
            device_id = input("\nEnter device number to use for recording (or press Enter for default): ").strip()
            
            if device_id == "":
                print("Using default input device")
                return None
            
            device_id = int(device_id)
            if 0 <= device_id < num_devices:
                device = sd.query_devices(device_id)
                if device['max_input_channels'] > 0:
                    print(f"Selected device: {device['name']}")
                    return device_id
                else:
                    print("This device doesn't support input. Please select another.")
            else:
                print("Invalid device number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def record_and_play(duration=5, samplerate=48000, device=None):
    """
    Record audio for specified duration and play it back.
    
    Args:
        duration (float): Recording duration in seconds
        samplerate (int): Sample rate in Hz
        device (int): Device ID to use for recording (None for default)
    """
    channels = 1
    
    # Set input device if specified
    if device is not None:
        sd.default.device[0] = device
        print(f"Using input device: {sd.query_devices(device)['name']}")
    
    print(f"\nRecording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype=np.float32
    )
    
    # Wait for recording to complete
    sd.wait()
    print("Recording finished!")
    
    print("Playing back recording...")
    # Play the recorded audio
    sd.play(recording, samplerate)
    sd.wait()  # Wait for playback to finish
    print("Playback finished!")

if __name__ == "__main__":
    try:
        # Let user select device
        device_id = select_device()
        
        # Get recording duration
        try:
            duration = float(input("\nEnter recording duration in seconds (press Enter for default 5s): ") or 5)
        except ValueError:
            print("Invalid duration. Using default 5 seconds.")
            duration = 5
        
        # Record and play test
        input("\nPress Enter to start recording...")
        record_and_play(duration=duration, device=device_id)
        
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")