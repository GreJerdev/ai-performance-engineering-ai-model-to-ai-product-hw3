from app.ai_agent.agent import start_agent
from dotenv import load_dotenv


def main():
    load_dotenv()
    start_agent()


if __name__ == "__main__":
    main()
