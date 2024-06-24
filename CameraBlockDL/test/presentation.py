import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
from copy import copy

file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

def make_confusion_matrix(total_ys, total_preds, cfg):
    cm = confusion_matrix(total_ys, total_preds)
    print("unnormalized cm")
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("normalized cm")
    print(cm)
    plot_cm(cfg, cm, custom_cm=False)


def plot_cm(cfg,confusion_mat, custom_cm=False):
    if hasattr(cfg.test, 'result_dir'):
        file_path = cfg.test.result_dir
    else:
        file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    confusion_mat_copy = np.nan_to_num(confusion_mat)
    print("confusion_mat_copy")
    print(confusion_mat_copy)
    accuracy = np.trace(confusion_mat_copy) / float(np.sum(confusion_mat_copy))
    misclass = 1 - accuracy
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('Blues')
    plt.imshow(confusion_mat, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix Traffic Light")
    plt.colorbar()
    plt.clim(0, 1)
    thresh = 0.8     # 0.8 이상이면 검은 글씨가 잘 안보여서 이렇게 수정
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[0])):
        plt.text(j, i, "{:0.3f}".format(confusion_mat[i, j]), horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    # plt.show()

    cid = cfg.id

    if custom_cm:
        plt.savefig(f"{file_path}/archived/custom_cm/{cid}_confusion_matrix.png")
    else:
        plt.savefig(f"{file_path}/archived/confusion_matrix/{cid}_confusion_matrix.png")


# calculate accuracy of each class
def calculate_each_class_accuracy(list_of_classes, preds, labels):
    acc = [0 for c in list_of_classes]
    for c in list_of_classes:
        acc[c] = ((preds == labels) * (labels == c)).float() / (max(labels == c).sum(), 1)
    return acc