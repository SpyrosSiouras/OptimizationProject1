from dataset import load_data, transform_data, plot_data

FILE_PATH = 'Government_Securities_GR.txt'


if __name__ == "__main__":

    data = load_data(FILE_PATH)

    data = transform_data(data)

    plot_data(data, 25)
