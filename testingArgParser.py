from argparse import ArgumentParser



def printphrase(phrase, nbaplayer):
    print(f"The phrase you entered is: {phrase}")
    print(f"I love: {nbaplayer}")


parser = ArgumentParser()
parser.add_argument('phrase', type=str, help='The phrase to print')
parser.add_argument('nbaplayer', type=str, help='nba player')


if __name__ == '__main__':
    args = parser.parse_args()
    args_dict = dict(args._get_kwargs())
    print(args_dict)
    printphrase(args.phrase, args.nbaplayer)
    printphrase(**args_dict)
