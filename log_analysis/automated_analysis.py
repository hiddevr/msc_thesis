import pm4py
import pandas as pd
import DyLoPro as dlp
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
import os
from tqdm import tqdm
import concurrent.futures


class NonInteractivePlot:
    def __enter__(self):
        self.original_backend = plt.get_backend()
        matplotlib_use('Agg')

    def __exit__(self, exc_type, exc_value, traceback):
        matplotlib_use(self.original_backend)


def load_data(file_path):
    el = pm4py.read_xes(file_path)
    return el


def find_xes_files(starting_folder):
    result = []

    for root, dirs, files in os.walk(starting_folder):
        for file in files:
            if file.endswith('.xes'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, starting_folder)
                parts = relative_path.split(os.sep)
                if len(parts) > 1:
                    dataset_name = parts[0]
                else:
                    dataset_name = os.path.splitext(file)[0]
                result.append({
                    'dataset_name': dataset_name,
                    'dataset_path': file_path
                })

    return result


def get_processed_datasets(save_folder):
    entries = os.listdir(save_folder)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(save_folder, entry))]
    return folders


def classify_columns(df: pd.DataFrame):
    categorical_case_features = []
    numeric_event_features = []
    categorical_event_features = []
    numeric_case_features = []

    case_columns = [col for col in df.columns if col.startswith('case:')]

    for col in case_columns:
        if col == 'case:concept:name' or df[col].nunique() < 2:
            continue
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            categorical_case_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_case_features.append(col)

    for col in df.columns:
        if col not in case_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_values = df.groupby('case:concept:name')[col].nunique()
                if unique_values.max() > 1:
                    numeric_event_features.append(col)
            elif df[col].dtype == 'object' or df[col].dtype == 'string':
                unique_values = df.groupby('case:concept:name')[col].nunique()
                if unique_values.max() > 1:
                    categorical_event_features.append(col)
    print(categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features)
    return categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features


def create_plot_obj(el, categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features):
    plot_object = dlp.DynamicLogPlots(el,
                                      'case:concept:name',
                                      'concept:name',
                                      'time:timestamp',
                                      categorical_case_features,
                                      numeric_case_features,
                                      categorical_event_features,
                                      numeric_event_features)
    return plot_object


def create_plots(plot_object, fig_save_folder, categorical_case_features, numeric_event_features,
                 categorical_event_features, numeric_case_features, dataset_name):
    with NonInteractivePlot():
        print(f'Generating plots for {dataset_name}...')
        print('Creating topK variants plot...')
        plot_object.topK_variants_evol()
        plt.savefig(os.path.join(fig_save_folder, 'topK_variants_evol.png'))
        plt.close()

        print('Creating topK dfr plot...')
        plot_object.topK_dfr_evol()
        plt.savefig(os.path.join(fig_save_folder, 'topK_dfr_evol.png'))
        plt.close()

        print('Creating categorical case feature evolution plots...')
        for attr in tqdm(categorical_case_features):
            plot_object.topK_categorical_caseftr_evol(attr, max_k=50)
            plt.savefig(os.path.join(fig_save_folder, f'topK_categorical_caseftr_evol_{str(attr).replace(":", "_")}.png'))
            plt.close()

        print('Creating numeric case feature evolution plot...')
        plot_object.num_casefts_evol(numeric_case_features, max_k=50)
        plt.savefig(os.path.join(fig_save_folder, f'numerical_caseftr_evol.png'))
        plt.close()

        print('Creating categorical event feature evolution plots...')
        for attr in tqdm(categorical_event_features):
            plot_object.topK_categorical_eventftr_evol(attr, max_k=50)
            plt.savefig(os.path.join(fig_save_folder, f'topK_categorical_eventftr_evol_{str(attr).replace(":", "_")}.png'))
            plt.close()

        print('Creating numeric event feature evolution plot...')
        plot_object.num_eventfts_evol(numeric_event_features, max_k=50)
        plt.savefig(os.path.join(fig_save_folder, f'numerical_eventftr_evol.png'))
        plt.close()


def process_file(file):
    el = load_data(file['dataset_path'])
    fig_save_folder = f'/local/s3377954/remote_ssh_files/log_analysis/automated_analysis/{file["dataset_name"]}'
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)

    categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features = classify_columns(el)
    plot_object = create_plot_obj(el, categorical_case_features, numeric_event_features, categorical_event_features, numeric_case_features)
    create_plots(plot_object, fig_save_folder, categorical_case_features, numeric_event_features,
                 categorical_event_features, numeric_case_features, file["dataset_name"])


def main():
    starting_folder = '/local/s3377954/remote_ssh_files/raw_eventlogs'
    xes_files = find_xes_files(starting_folder)
    processed_datasets = get_processed_datasets('/local/s3377954/remote_ssh_files/log_analysis/automated_analysis')
    xes_files = [file for file in xes_files if file['dataset_name'] not in processed_datasets]

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(process_file, xes_files), total=len(xes_files)))


if __name__ == '__main__':
    main()
