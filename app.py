from agent import SimpleEchoAgent
import mlflow
from mlflow.genai.serving import AgentServer

def main():
    mlflow.models.set_model(SimpleEchoAgent())
    AgentServer().serve(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
