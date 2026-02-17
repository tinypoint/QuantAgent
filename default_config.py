DEFAULT_CONFIG = {
    "agent_llm_model": "gpt-5.3-codex",
    "graph_llm_model": "gpt-5.3-codex",
    "agent_llm_provider": "openai-codex",  # "openai", "openai-codex", "anthropic", or "qwen"
    "graph_llm_provider": "openai-codex",  # "openai", "openai-codex", "anthropic", or "qwen"
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "sk-",  # OpenAI API key
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
}
