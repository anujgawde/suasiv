from suasiv.llm.base import LLMBackend
from suasiv.llm.stub import StubBackend

BACKENDS: dict[str, type[LLMBackend]] = {
    "stub": StubBackend,
}


def get_backend(name: str) -> LLMBackend:
    cls = BACKENDS.get(name)
    if cls is None:
        return StubBackend()
    return cls()
