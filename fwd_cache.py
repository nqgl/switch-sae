from typing import Any


class FwdCache:
    def __init__(self):
        self._cache = {}

    def __getattribute__(self, __name: str) -> Any:
        if __name == "_cache":
            return object.__getattribute__(self, __name)
        else:
            return self._cache[__name]

    def __setattr__(self, __name: str, value: Any) -> None:
        if __name == "_cache":
            object.__setattr__(self, __name, value)
        else:
            assert (
                __name not in self._cache
            ), f"__name={__name} already exists in cache!"
            self._cache[__name] = value
