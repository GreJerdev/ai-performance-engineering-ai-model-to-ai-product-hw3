import os

class LLMBase:

    BASE_URL="https://api.tokenfactory.uk-south1.nebius.com/v1/"
    API_KEY=os.getenv("NEBIUS_API_KEY")
    def __init__(self):
        pass