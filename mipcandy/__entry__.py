from argparse import ArgumentParser

from mipcandy.run import config


def __entry__() -> None:
    parser = ArgumentParser(prog="MIP Candy CLI", description="MIP Candy Command Line Interface",
                            epilog="GitHub: https://github.com/ProjectNeura/MIPCandy")
    parser.add_argument("-c", "--config", choices=("setting", "secret"), default=None,
                        help="set a configuration such that key=value")
    parser.add_argument("-kv", "--key-value", nargs=2, action="append", default=None, help="define a key-value pair")
    args = parser.parse_args()
    if args.config:
        if not args.key_value:
            raise ValueError("Expected at least one key-value pair")
        for key_value in args.key_value:
            config(args.config, key_value[0], key_value[1])
