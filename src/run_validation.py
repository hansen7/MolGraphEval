from config.validation_config import parse_config
from init import get_device, get_task, init
from validation.task import Task


def main() -> None:
    config = parse_config()
    init(config=config)
    task = get_task(config=config)
    task.run()


if __name__ == "__main__":
    main()
