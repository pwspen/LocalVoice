# Features
- Interrupt model at any time
- Typical end of user speech to start of assistant speech latency: ~1s
    - With 100 tok/s local model, no text->tts or tts->audio streaming
- Kokoro, Zonos, or Llasa for STT, Whisper for TTS, Silero for VAD, any OpenAI API for LLM (Ollama, Openrouter, etc) - but easily add any model!

Example use:
```
from localvoice import KokoroTTS, WhisperSTT, LLM, VoiceAssistant, TTSBase, STTBase
from utils import select_device

# Add any TTS or STT by making classes that inherit from TTSBase or STTBase and implement 
# transcribe(audio: np.ndarray) -> str, or,
# speak(sentence: str) -> np.ndarray

tts = KokoroTTS()
# or:
# tts = ZonosTTS(condfile="skulls.wav")
# tts = LlasaTTS(condfile="skulls.wav")

stt = WhisperSTT()

llm = LLM(provider="ollama", model="llama3.1:8b")
# or:
# llm = LLMProcessor(
#     provider="openrouter",
#     model="anthropic/claude-3.5-sonnet",
#     api_key=os.getenv("OPENROUTER_API_KEY")
# )

va = VoiceAssistant(
    tts = tts,
    stt = stt,
    llm = llm,
    input_device = select_device()
)

va.active()
```


# TODO
- [x] TTS sentence chunking
- [x] Interrupt detection
- [ ] Tool use (PydanticAI)
- [x] LLaSa integration (any voice, any persona)
- [x] Zonos integration
- [ ] vision LLM / YOLO integration
- [ ] Biasing voice with emotion
- [ ] OpenHands CLI integration
- [ ] Life / goals coach module

- [ ] Wake word, conversation end modes
- [ ] text -> tts streaming
- [ ] tts -> audio streaming (in chunks for now)
- [ ] Avg latency auto tests
- [ ] Audio preprocessing/filtering
- [ ] Visual interface - openwebui?
- [ ] ez well tested install script

VAD + Whisper + Kokoro + Llama 8b is only 10 gigs, llama 8b being 5 of them. There's definitely room for LLasa plus whatever other fun models.

The goal of this project is to make it easy to run a full voice + LLM pipeline locally on GPU, and make it simple enough so tinkering with it is also easy. Then to be a model organism for an agent interface. 

Install should be single script, no manually downloading anything

As long as they are optional and fun/interesting/useful, just keep piling on more models and tools