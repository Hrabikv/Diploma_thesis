from preprocessing.datasets_loaders import Kodera_29, Averta_156

if __name__ == '__main__':
    # pom = Averta_156.Averta()

    pom = Kodera_29.Kodera()
    files_per_subject = pom.load_data()
    print()
