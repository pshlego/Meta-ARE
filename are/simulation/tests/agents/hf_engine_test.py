from are.simulation.agents.llm.hf import hf_engine


def test_openai_provider_uses_openai_api_key(monkeypatch):
    captured = {}

    class FakeInferenceClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("HF_TOKEN", "hf-key")
    monkeypatch.setattr(hf_engine, "InferenceClient", FakeInferenceClient)

    hf_engine.HuggingFaceLLMEngine(
        hf_engine.HuggingFaceModelConfig(
            model_name="gpt-4.1-mini",
            provider="openai",
        )
    )

    assert captured["api_key"] == "openai-key"