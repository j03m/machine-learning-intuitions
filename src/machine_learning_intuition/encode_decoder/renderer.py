import matplotlib.pyplot as plt


class TileRenderer:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def plot(self, labels, x, y, tile):
        plt.subplot(self.rows, self.cols, tile)

        plt.scatter(x, y,
                    s=10, alpha=0.8, cmap='Set1', c=labels)
        plt.xlabel('D1')
        plt.ylabel('D2')

    def tiles(self):
        return self.rows * self.cols

    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def save(self, path):
        plt.savefig(path)
