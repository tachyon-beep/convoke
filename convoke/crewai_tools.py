class BaseTool:
    """Stub BaseTool to support scoped artifact tools."""

    def __init__(self, *args, **kwargs):
        # Store any passed attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.logger = kwargs.get("store").logger if "store" in kwargs else None

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass.")

    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)
