'''
To run this, cd into Dynamic and input the following into the command line:
python testing/testingArgParser.py "hello" "LebronJames"

Your output:
The phrase you entered is: hello
I love: LebronJames

'''

from argparse import ArgumentParser


def printphrase(phrase, nbaplayer, save_video, num_epochs):
    print(f"The phrase you entered is: {phrase}")
    print(f"I love: {nbaplayer}")

    if save_video == True: 
        print ("save_video is a parameter")
    else:
        print("ERROR: SAVE_VIDEO ISN'T A PARAMETER")

    print(f" The num_epochs = {num_epochs}")




if __name__ == '__main__':
    #This is standard set up of the CLI 
    # args = parser.parse_args() 
    # printphrase(args.phrase, args.nbaplayer) 

    #if you want you can store arguments in a dictionary like so 
    # args_dict = dict(args._get_kwargs()) 

    #view the dictionary like so 
    # print(args_dict)
    #if you call the same 'python testingArgParser.py "hello" "LebronJames"' it will print the same expected output but this time using a dictionary. This is useful for many arguments. 
    # printphrase(**args_dict)

    #For testing of the save_video_group 
    # print(args_dict) #should return you {'phrase': 'hello', 'nbaplayer': 'LebronJames', 'save_video': False} when you input 'python testing/testingArgParser.py "hello" "LebronJames"' (note I changed directory structure)
    #print(args_dict) # will make save_video: True in the above dict if input is: python testing/testingArgParser.py "hello" "LebronJames" --save-video

    #I have modified printphrase to also take --save-video as an argument, but note that argparser seems to automatically convert '--save-video' into a parameter "save_video"
    #printphrase(**args_dict) #An input with the --save-video will give you an error

    #checking --num-epochs
    #input: python testing/testingArgParser.py "hello" "LebronJames" --save-video --num-epochs=5
    # print(args_dict)
    # printphrase(**args_dict) 
    
    # INPUTS AND OUTPUTS IN CLI 
    # PS C:\Users\Allis\Documents\MDN\Ultrasound2023\dynamic> python testing/testingArgParser.py "hello" "LebronJames" --save-video --num-epochs=5
    # {'phrase': 'hello', 'nbaplayer': 'LebronJames', 'save_video': True, 'num_epochs': 5}
    # The phrase you entered is: hello
    # I love: LebronJames
    # save_video is a parameter
    #  The num_epochs = 5
    


    #testing input(specifically the -m parts): python -m testing/testingArgParser.py "hello" "LebronJames" --save-video --num-epochs=5
    #your bug was that you needed to move the arguments into the if==name part of the file
    #you also have to change the input to this: python -m testing.testingArgParser "hello" "LebronJames" --save-video --num-epochs=5 
    parser = ArgumentParser()
    parser.add_argument('phrase', type=str, help='The phrase to print')
    parser.add_argument('nbaplayer', type=str, help='nba player')

    #--save-video argument 
    save_video_group = parser.add_mutually_exclusive_group()
    save_video_group.add_argument("--save-video", action="store_true", default=False)

    #num epochs argument
    parser.add_argument("--num-epochs", type=int, default=50)
    
    args = parser.parse_args() 
    args_dict = dict(args._get_kwargs()) 

    print(args_dict)
    printphrase(**args_dict)  