import os

def count_files_in_images_folder(folder_path):
    total_file_count = 0
    excluded_folders = {'belgium', 'spain', 'sweden', 'sweden', 'germany'}  # 要排除的子文件夹名称

    for subdir in os.listdir(folder_path):
        if subdir in excluded_folders:
            continue  # 跳过特定名称的子文件夹

        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            images_folder = os.path.join(subdir_path, 'images')
            if os.path.exists(images_folder):
                file_count = len(os.listdir(images_folder))
                total_file_count += file_count
    return total_file_count

if __name__ == "__main__":
    wrcg_data_path = '../wrcg_data'  # 请根据实际路径修改
    total_files = count_files_in_images_folder(wrcg_data_path)
    print(f"总文件数: {total_files}")
