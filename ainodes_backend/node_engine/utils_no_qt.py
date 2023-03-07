import traceback
from pprint import PrettyPrinter


def dumpException(e=None):
    """
    Prints out an Exception message with a traceback to the console

    :param e: Exception to print out
    :type e: Exception
    """
    # print("%s EXCEPTION:" % e.__class__.__name__, e)
    # traceback.print_tb(e.__traceback__)
    traceback.print_exc()


pp = PrettyPrinter(indent=4).pprint

