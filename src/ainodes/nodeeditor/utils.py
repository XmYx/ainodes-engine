# -*- encoding: utf-8 -*-
"""
Module with some helper functions
"""
from qtpy.QtCore import QFile
from qtpy.QtWidgets import QApplication

from nodeeditor.utils_no_qt import pp, dumpException


def loadStylesheet(filename: str):
    """
    Loads an qss stylesheet to the current QApplication instance

    :param filename: Filename of qss stylesheet
    :type filename: str
    """
    print('STYLE loading:', filename)
    file = QFile(filename)
    file.open(QFile.ReadOnly | QFile.Text)
    stylesheet = file.readAll()
    QApplication.instance().setStyleSheet(str(stylesheet, encoding='utf-8'))

def loadStylesheets(*args):
    """
    Loads multiple qss stylesheets. Concatenates them together and applies the final stylesheet to the current QApplication instance

    :param args: variable number of filenames of qss stylesheets
    :type args: str, str,...
    """
    res = ''
    for arg in args:
        file = QFile(arg)
        file.open(QFile.ReadOnly | QFile.Text)
        stylesheet = file.readAll()
        res += "\n" + str(stylesheet, encoding='utf-8')
    QApplication.instance().setStyleSheet(res)
