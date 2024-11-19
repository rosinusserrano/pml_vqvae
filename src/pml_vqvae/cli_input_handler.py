import argparse

from pml_vqvae.config_class import Config


class CLI_handler:
    """Command line interface handler for maximum flexibility in desinging experiments"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--name",
            "-n",
            help="name of the experiment. If not provided, the name from the config file will be used",
        )
        self.parser.add_argument(
            "--description",
            "-d",
            help="description of the experiment. If not provided, the description from the config file will be used",
        )
        self.parser.add_argument(
            "--seed",
            "-s",
            type=int,
            help="seed for reproducibility. If not provided, the seed from the config file will be used",
        )
        self.parser.add_argument(
            "--n_train",
            help="number of training samples. If not provided, the number from the config file will be used",
            type=int,
        )
        self.parser.add_argument(
            "--n_test",
            help="number of test samples. If not provided, the number from the config file will be used",
            type=int,
        )
        self.parser.add_argument(
            "--batch_size",
            "-bs",
            help="batch size for training. If not provided, the batch size from the config file will be used",
            type=int,
        )
        self.parser.add_argument(
            "--epochs",
            "-e",
            help="number of epochs for training. If not provided, the number from the config file will be used",
            type=int,
        )
        self.parser.add_argument(
            "--learning_rate",
            "-lr",
            help="learning rate for training. If not provided, the learning rate from the config file will be used",
            type=float,
        )
        self.parser.add_argument(
            "--model_name",
            "-mn",
            help="name of the model to train. If not provided, the model name from the config file will be used",
        )
        self.parser.add_argument(
            "--dataset",
            "-ds",
            help="name of the dataset to use. If not provided, the dataset name from the config file will be used",
        )
        self.parser.add_argument(
            "--test_interval",
            "-ti",
            type=int,
            help="interval for testing the model. If not provided, the interval from the config file will be used",
        )
        self.parser.add_argument(
            "--vis_train_interval",
            "-vti",
            type=int,
            help="interval for visualizing the training process. If not provided, the interval from the config file will be used",
        )
        self.parser.add_argument(
            "--wandb_log",
            "-wl",
            type=bool,
            help="log results to wandb. If not provided, the wandb log from the config file will be used",
        )

    def parse_args(self):
        """Parse command line arguments

        Returns:
            argparse.Namespace: Command line arguments
        """

        args = self.parser.parse_args()
        return args

    def adjust_config(self, config: Config, args: argparse.Namespace):
        """Adjust the configuration object based on the command line arguments

        Args:
            config (Config): Configuration object
            args (argparse.Namespace): Command line arguments

        Returns:
            Config: Adjusted configuration object
        """

        if args.name:
            config.name = args.name
        if args.description:
            config.description = args.description
        if args.seed:
            config.seed = args.seed
        if args.n_train:
            config.n_train = args.n_train
        if args.n_test:
            config.n_test = args.n_test
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.epochs:
            config.epochs = args.epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.model_name:
            config.model_name = args.model_name
        if args.dataset:
            config.dataset = args.dataset
        if args.test_interval:
            config.test_interval = args.test_interval
        if args.vis_train_interval:
            config.vis_train_interval = args.vis_train_interval
        if args.wandb_log:
            config.wandb_log = args.wandb_log

        return config
