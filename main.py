import yaml
from pipeline.runner import TradingPipeline

def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/config.yaml")
    pipeline = TradingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()