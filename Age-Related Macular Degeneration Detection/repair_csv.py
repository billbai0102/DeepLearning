"""
repair the csv since it's missing a few values :(
"""
import os
import pandas as pd

if __name__ == "__main__":
    path = './data/Training400/AMD'
    dir_len = len(os.listdir(path))

    # fix filenames
    for idx, file in enumerate(os.listdir(path)):
        current_file = os.path.join(path, file)
        new_filename = os.path.join(path, f'A00{idx+1:02d}.jpg')
        os.rename(current_file, new_filename)

    # fix csv
    targets_path = './data/Training400/Fovea_location.xlsx'
    targets_csv = pd.read_excel(targets_path)
    targets_csv['ID'] = [i for i in range(1, len(targets_csv) + 1)]
    armd_csv = targets_csv[:dir_len]
    armd_csv['imgName'] = [f'A00{idx:02d}.jpg' for idx in range(1, 88)]
    print(targets_csv)
    targets_csv.to_excel(targets_path)
