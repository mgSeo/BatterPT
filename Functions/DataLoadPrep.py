import pandas as pd


def __init():
    return 0

def load_data(folder_path, file_path, file_sheet):

    # Read all sheets at once
    data = pd.read_excel(folder_path + file_path, sheet_name=file_sheet, engine='openpyxl')

    return data
