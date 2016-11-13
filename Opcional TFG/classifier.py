import pickle as pick


def train_GaussianBayes():
    print("TO-DO")


def sing_classifier(sign):
    print("TO-DO")


def check_trainDescriptor():
    print("TO-DO")


def saveToDisk(object, fileName):
    print("SAVING...")
    file = fileName + ".pickle"
    with open(fileName, 'wb') as handle:
        pick.dump(object, handle)


def loadFromDisk(fileName):
    print("LOADING...")
    file = fileName + ".pickle"
    with open(fileName, 'wb') as handle:
        dst = pick.load(handle)
        return dst
    return None
