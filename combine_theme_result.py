import os
import pandas as pd

def combine_theme_result():
    folders = ['res_pure', 'res_wnamespace', 'res_wdependency']
    files = os.listdir(folders[0])

    dir_output = 'res_combined'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    for file in files:
        if file.endswith('.csv'):
            data_frames = []
            for i, folder in enumerate(folders):
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)
                if i != 0:
                    df = df.iloc[:, 1:]
                data_frames.append(df)
            result = pd.concat(data_frames, axis=1)
            result.to_excel(os.path.join(dir_output, file.replace('.csv', '.xlsx')), index=False)


if __name__ == "__main__":
    combine_theme_result()
