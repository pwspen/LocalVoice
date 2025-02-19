# Features
- Interrupt model at any time
- Typical end of user speech to start of assistant speech latency: ~1s
    - With 100 tok/s local model, no text->tts or tts->audio streaming
- Kokoro or Zonos for STT, Whisper for TTS, Silero for VAD, any OpenAI API for LLM (Ollama, Openrouter, etc)


# TODO
- [x] TTS sentence chunking
- [x] Interrupt detection
- [ ] Tool use (PydanticAI)
- [ ] LLaSa integration (any voice, any persona)
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

Install should be single script, that also optionally tries to take care of cuda deps. No git lfs

As long as they are optional and fun/interesting/useful, just keep piling on more models and tools