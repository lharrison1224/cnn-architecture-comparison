import load_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    train, train_labels, test, test_labels = load_data.load()
    print(np.shape(train))
    print(np.shape(test))
    plt.imshow(test[20])
    plt.show()


if __name__ == "__main__":
    main()
