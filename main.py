from preprocessing import ETL
from vizualization.Raw_data_vizualization import plot_data_of_subject, plot_data_of_all_subject

if __name__ == '__main__':
    data, labels = ETL.load_data()
    plot_data_of_all_subject(data, labels, "")
    print()
