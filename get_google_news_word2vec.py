import gensim
import numpy as np
# load google pre-train word2vec embedding
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

# cifar 100 dataset labels
def get_label_embedding():
    CIFAR100_LABELS_LIST = [
        'apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear'
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    maxtrix = np.zeros((100, 300))
    maxtrix = np.reshape(maxtrix,(100,300))
    # get label embedding
    string = model[str(CIFAR100_LABELS_LIST[0])]
    string = string.reshape(300,1)
    print(string.shape)
    # save label embedding as label_embedding.npy
    for i in range(100):
        string = model[str(CIFAR100_LABELS_LIST[i])]
        maxtrix[i] = string
    maxtrix = np.reshape(maxtrix,(100*300))
    np.save('label_embedding.npy',maxtrix)

# Zero shot label embedding
def get_zero_shot_label_embedding():
    CIFAR100_LABELS_LIST = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
        'ship', 'truck','apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    maxtrix = np.zeros((110, 300))
    maxtrix = np.reshape(maxtrix,(110,300))
    # get label embedding
    string = model[str(CIFAR100_LABELS_LIST[0])]
    string = string.reshape(300,1)
    print(string.shape)

    for i in range(110):
        string = model[str(CIFAR100_LABELS_LIST[i])]
        maxtrix[i] = string
    maxtrix = np.reshape(maxtrix,(110*300))
    np.save('zero_shot_label_embedding.npy',maxtrix)

    label_embedding = np.load('zero_shot_label_embedding.npy')
    print(label_embedding.shape)

if __name__ == "__main__":
    get_label_embedding()
    get_zero_shot_label_embedding()