import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser(description='tsne instruction and visualization')
    parser.add_argument('-perplexity', default=10)
    args = parser.parse_args()
    return args


def tsne(all_fea):
    """
    :param all_fea: data   nxm
    :return: a array which is samples x 2 or 3
    """

    # n_components is descending into 2 dimensions
    # perplexity is a guessing: maybe one class has 10 samples
    tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    X_d = tsne.fit_transform(all_fea)
    # note: you get X_2d is samples x 2 array.
    return X_d


def draw(x,label_state,label,classifcation):
    """
    :param x: you data which need to visualization
    :param label_state: yes or no,if you have labels, you should input yes
    :param label: your labels    samples x 1
    :param classifcation: how many class do you have   a number
    :return:none
    """
    if label_state == 'no':
        plt.scatter(x[:,0],x[:,1],c='r')
    # we have label like:  a->14   means  a is 14 class
    if label_state == 'yes':
        color = []
        # so we need to get some color to description the different dot

        for name, hex in matplotlib.colors.cnames.items():
            if len(color) < classifcation:
                color.append(name)
        # draw
        for i,pairs in enumerate(x):
            label_ = label[i][0]
            # c is color of label of i-th sample
            plt.scatter(pairs[0],pairs[1],c=color[label_])
    plt.show()


def start():
    # here is your data, tensor/array/list ...
    all_fea = torch.randn(500,10)    # samples  and  dimensions
    label_state = 'yes'
    class_number = 10
    label = np.random.randint(0,class_number,[500,1])

    # tsne
    fea = tsne(all_fea)

    # draw
    draw(fea,label_state,label,class_number)


if __name__ == '__main__':
    start()
