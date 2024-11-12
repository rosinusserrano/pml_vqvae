import argparse

from pml_vqvae.config_class import Config


class CLI_handler:
    """Command line interface handler"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--name",
            help="name of the experiment. If not provided, the name from the config file will be used",
        )
        self.parser.add_argument(
            "--description",
            help="description of the experiment. If not provided, the description from the config file will be used",
        )
        self.parser.add_argument(
            "--seed",
            help="seed for reproducibility. If not provided, the seed from the config file will be used",
        )
        self.parser.add_argument(
            "--n_train",
            help="number of training samples. If not provided, the number from the config file will be used",
        )
        self.parser.add_argument(
            "--n_test",
            help="number of test samples. If not provided, the number from the config file will be used",
        )
        self.parser.add_argument(
            "--batch_size",
            help="batch size for training. If not provided, the batch size from the config file will be used",
        )
        self.parser.add_argument(
            "--epochs",
            help="number of epochs for training. If not provided, the number from the config file will be used",
        )
        self.parser.add_argument(
            "--learning_rate",
            help="learning rate for training. If not provided, the learning rate from the config file will be used",
        )
        self.parser.add_argument(
            "--model_name",
            help="name of the model to train. If not provided, the model name from the config file will be used",
        )
        self.parser.add_argument(
            "--dataset",
            help="name of the dataset to use. If not provided, the dataset name from the config file will be used",
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

        return config
