#=== GET_FILE_PATH.py script ===#
import os


def get_data_file_path(Ymd_str, Ym_str, Y_str, data_path="prediction_data"):
    """Gets the file path of the data directory
    Args:
    cwd - gets the file path of the current working directory
    parent - the folder name of the parent directory
    """

    cwd = os.getcwd()
    parent = os.path.basename(os.path.dirname(cwd))
    if os.path.basename(cwd) == "my_app":
        return f"{cwd}/data/result/{data_path}/{Y_str}/{Ym_str}/{Ymd_str}/"
    elif parent == "outputs":
        return f"{os.path.dirname(os.path.dirname(cwd))}/data/result/{data_path}/{Y_str}/{Ym_str}/{Ymd_str}/"
    elif parent == "my_app":
        return f"{os.path.dirname(cwd)}/data/result/{data_path}/{Y_str}/{Ym_str}/{Ymd_str}/"
    return f"Error - check file path: {os.path.dirname(cwd)}/data/result/{data_path}/{Y_str}/{Ym_str}/{Ymd_str}/"