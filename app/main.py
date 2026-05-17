from data_layer.bitext_data_layer import BitextDataLayer


def main():
    data_layer = BitextDataLayer("bitext_data.csv")
    data_layer.get_data()


if __name__ == "__main__":
    main()