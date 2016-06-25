import glob
import json
from nltk import FreqDist, word_tokenize

all_tokens = []

def loadFile(filename):
    with open(filename) as file:
        data = json.load(file)
    return data

def processMessage(message):
    if 'subtype' not in message and message['type'] == 'message':
        return word_tokenize(message['text'])

def processMessages(messages):
    for message in messages:
        tokens = processMessage(message)
        if (tokens != None):
            all_tokens.extend(tokens)

def main():
    for file in glob.glob('./*/*/*.json'):
        data = loadFile(file)
        processMessages(data)

    fdist = FreqDist(all_tokens)
    # Output top 50 words
    for word, frequency in fdist.most_common(50):
        print('%s;%d' % (word, frequency)).encode('utf-8')
main()
