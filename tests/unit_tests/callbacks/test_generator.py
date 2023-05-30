from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import FakeListLLM
from langchain.callbacks import GeneratorCallbackHandler


def test_generator_callback() -> None:
    handler = GeneratorCallbackHandler()

    llm = FakeListLLM(
        responses=[
            "I need to search for LangChain\nAction: Wikipedia\nAction Input: LangChain",
            "I am done\nFinal Answer: LangChain is ...",
        ]
    )

    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    agent.run("What is LangChain", callbacks=[handler])

    CHAIN = handler._CHAIN
    LLM = handler._LLM
    TOOL = handler._TOOL

    expected_texts = [
        ("Entering new LLMChain chain", CHAIN),
        ("Prompt after formatting", LLM),
        ("Finished chain", CHAIN),
        ("I need to search for LangChain", TOOL),
        ("Observation", TOOL),
        ("Entering new LLMChain chain", CHAIN),
        ("Prompt after formatting", LLM),
        ("Finished chain", CHAIN),
        ("I am done", TOOL),
        ("Finished chain", CHAIN),
    ]

    for expected_text, expected_source in expected_texts:
        text, source = next(handler)
        assert expected_text in text
        assert expected_source == source
