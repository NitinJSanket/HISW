#-----------------------------------------------------------------------
#IMPORTS
from __future__ import division
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from nltk import word_tokenize
from nltk import pos_tag, map_tag
import sys
import os

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#UTIL FUNCTIONS
def loadFileLines(filepath):
    fileInstance = open(filepath)
    fileLines = []
    while True:
        line = fileInstance.readline()
        if not line:
            fileInstance.close()
            break

        fileLines.append(line.split(" ")[1][:-1])
    return fileLines


def saveListToFile(fileName, listName):
    outputFile = open(fileName,'wb')

    for key in listName:
        outputFile.write(str(key))
        outputFile.write('\n')

    outputFile.close()
    
    return True

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#MAIN - TESTING THE CODE FROM MAIN

if __name__ == "__main__":
    print("=====ASSIGNMENT STARTED=====")

    """
    fileName = "G:\\Academics\\Fall2015\\ML520\\Project\\kit\\kit\\allWords.txt"
    allWords = loadFileLines(fileName)

    posTags = pos_tag(allWords)
    
    
    mapper = dict([('ADJ', 1), ('ADP', 2), ('ADV', 3), ('CONJ', 4), ('DET', 5), ('NOUN', 6), ('NUM', 7), ('PRT', 8), ('PRON', 9), ('VERB', 10), ('.', 11), ('X', 12)])
    simplifiedTags = [mapper[map_tag('en-ptb', 'universal', tag)] for word, tag in posTags]

    fileName = "G:\\Academics\\Fall2015\\ML520\\Project\\kit\\kit\\posTags.txt"
    saveListToFile(fileName, simplifiedTags)
    """
    
    fileName = "G:\\Academics\\Fall2015\\ML520\\Project\\kit\\kit\\allWords.txt"
    allWords = loadFileLines(fileName)
    wordsLen = []

    for word in allWords:
        wordsLen.append(len(word))

    fileName = "G:\\Academics\\Fall2015\\ML520\\Project\\kit\\kit\\wordLengths.txt"
    saveListToFile(fileName, wordsLen)
        
