class CustomException(Exception):
    def __init__(self, errorinfo):
        super().__init__(errorinfo)
