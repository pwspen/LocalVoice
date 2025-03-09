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