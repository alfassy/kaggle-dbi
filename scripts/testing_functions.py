import numpy as np
import torch


def calc_accuracy(outputs, targets):
    '''
    Calculates accuracy
    :param outputs: np array
    :param targets: np array
    :return: float
    '''
    outputs_max = np.argmax(outputs, axis=1)
    accuracy = (outputs_max == targets).mean()
    return accuracy

def main():
    # filePath = 'C:/Users/Alfassy/PycharmProjects/oxford_dogs_and_cats/Cats-and-Dogs-Classification/yo.txt'
    # filePath = 'C:/Users/Alfassy/PycharmProjects/Dog Breed Identification/saved_models/test.pkl'
    filePath = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/inception_trainDBIInceptionHalf2019324153epoch;0.pkl'
    a = np.array([1,2,3])
    with open(filePath, 'wb') as f:
        torch.save(a, f)


if __name__ == '__main__':
    main()
