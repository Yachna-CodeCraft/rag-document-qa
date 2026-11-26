from app import config


def get_llm():
    """Return a chat model. Provider is chosen with LLM_PROVIDER in .env."""
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0)
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=0, max_tokens=1024)