from argparse import ArgumentParser, Namespace
import safitty
import pytorch_lightning as pl

from src import LightningModule


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("-t", "--trainer", required=True, type=str)

    return parser.parse_args()


def read_config(path: str) -> Namespace:
    config = safitty.load(path)
    return Namespace(**config)


def main():
    args = parse_args()

    config = read_config(args.config)
    trainer_config = read_config(args.trainer)

    model = LightningModule(params_=config)
    trainer = pl.Trainer.from_argparse_args(trainer_config,)
    trainer.fit(model)


if __name__ == "__main__":
    main()
