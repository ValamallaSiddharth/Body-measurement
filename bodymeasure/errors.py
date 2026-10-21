"""Public input/capture error shared by all entry points."""
class CaptureError(Exception):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


