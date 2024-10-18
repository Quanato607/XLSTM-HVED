import os


def write_foldernames_to_file(folder_path, output_file):
    with open(output_file, 'w') as f:
        # 只遍历一级目录
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                f.write(dir_name + '\n')
            # 只遍历一次，所以在找到一级目录后立即退出
            break

# 使用示例
folder_path = '/root/autodl-tmp/BraTS2024/test'  # 指定要读取的文件夹路径
output_file = 'check.txt'  # 指定要输出的文档路径
write_foldernames_to_file(folder_path, output_file)
