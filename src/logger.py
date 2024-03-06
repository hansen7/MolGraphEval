# Ref: https://raw.githubusercontent.com/davda54/sam/main/example/utility/log.py

import time, wandb, logging, dataclasses
from pathlib import Path
from datetime import datetime, timedelta
from config.training_config import TrainingConfig

TIME_STR = "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())


@dataclasses.dataclass
class EpochState:
    loss: float = 0.0
    accuracy: float = 0.0
    samples: int = 0

    def reset(self) -> None:
        self.loss, self.accuracy, self.samples = 0.0, 0.0, 0

    def add_to_loss(self, loss: float) -> None:
        self.loss += loss

    def add_to_accuracy(self, accuracy: float) -> None:
        self.accuracy += accuracy

    def add_to_samples(self, samples: int) -> None:
        self.samples += samples


class LoadingBar:
    def __init__(self, length: int = 40) -> None:
        self.length = length
        self.symbols = ["┈", "░", "▒", "▓"]

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length * 4 + 0.5)
        d, r = p // 4, p % 4
        return (
            "┠┈"
            + d * "█"
            + (
                (self.symbols[r]) + max(0, self.length - 1 - d) * "┈"
                if p < self.length * 4
                else ""
            )
            + "┈┨"
        )


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_info_logger(filepath) -> logging.Logger:
    """
    Create a logger.
    """

    Path("log").mkdir(parents=True, exist_ok=True)
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # TODO: If we want to print logs into the console, we can un-comment this
    # create console handler and set level to info
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setLevel(0)
    # console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


@dataclasses.dataclass
class CombinedLogger:
    config: TrainingConfig
    is_train: bool = True
    epoch_state: EpochState = EpochState()
    last_steps_state: EpochState = EpochState()
    loading_bar: LoadingBar = LoadingBar(length=27)
    best_val_accuracy: float = 0.0
    epoch: int = -1
    step: int = 0
    info_logger: logging.Logger = None

    def __post_init__(self):
        self.info_logger = create_info_logger(
            f"{self.config.log_filepath}-{TIME_STR}.log"
        )

    def log_value_dict(self, value_dict: dict):
        if self.config.log_to_wandb:
            wandb.log(
                {
                    "epoch": self.epoch,
                    **value_dict,
                }
            )
        self.info_logger.info(value_dict)

    def train(self, num_batches: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()
        self.is_train = True
        self.last_steps_state.reset()
        self._reset(num_batches)

    def eval(self, num_batches: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(num_batches)

    def __call__(
        self, loss, accuracy, batch_size: int, learning_rate: float = None
    ) -> None:
        # TODO: in training, we don't have to record the accuracy
        if self.is_train:
            self._train_step(loss, accuracy, batch_size, learning_rate)
        else:
            self._eval_step(loss, accuracy, batch_size)

    def flush(self) -> None:
        loss = self.epoch_state.loss / self.num_batches
        accuracy = self.epoch_state.accuracy / self.epoch_state.samples
        if self.is_train:
            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100 * accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )
            train_statistics = {
                "epoch": self.epoch,
                "train_accuracy": accuracy,
                "train_loss": loss,
                "lr": self.learning_rate,
            }
            if self.config.log_to_wandb:
                wandb.log(train_statistics)
            self.info_logger.info(train_statistics)

        else:
            print(f"{loss:12.4f}  │{100 * accuracy:10.2f} %  ┃", flush=True)

            if accuracy > self.best_val_accuracy:
                self.best_val_accuracy = accuracy
            validation_statistics = {
                "epoch": self.epoch,
                "val_accuracy": accuracy,
                "val_loss": loss,
                "best_val_accuracy": self.best_val_accuracy,
            }
            if self.config.log_to_wandb:
                wandb.log(validation_statistics)
            self.info_logger.info(validation_statistics)

    def _train_step(
        self, loss: float, accuracy: float, batch_size: int, learning_rate: float
    ) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state.add_to_loss(loss)
        self.last_steps_state.add_to_accuracy(accuracy)
        self.last_steps_state.add_to_samples(batch_size)
        self.epoch_state.add_to_loss(loss)
        self.epoch_state.add_to_accuracy(accuracy)
        self.epoch_state.add_to_samples(batch_size)
        self.step += 1

        if self.step % self.config.log_interval == self.config.log_interval - 1:
            loss = self.last_steps_state.loss / self.step
            accuracy = self.last_steps_state.accuracy / self.last_steps_state.samples

            self.last_steps_state.reset()
            progress = self.step / self.num_batches
            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100 * accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss: float, accuracy: float, batch_size: int) -> None:
        self.epoch_state.add_to_loss(loss)
        self.epoch_state.add_to_accuracy(accuracy)
        self.epoch_state.add_to_samples(batch_size)

    def _reset(self, num_batches: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.num_batches = num_batches
        self.epoch_state.reset()

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(
            "┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓"
        )
        print(
            "┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃"
        )
        print(
            "┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃"
        )
        print(
            "┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨"
        )
