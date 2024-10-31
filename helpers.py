import os

def get_textbooks(dataset_path):
    textbooks = []
    for book in os.listdir(dataset_path + 'textbooks/en/'):
        file = open(dataset_path + 'textbooks/en/' + book, 'r')
        textbooks.append('\n'.join(file.readlines()))

    return textbooks
