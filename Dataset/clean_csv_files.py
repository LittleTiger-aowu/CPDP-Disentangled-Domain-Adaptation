import pandas as pd
import os
import glob
import csv

def clean_csv_file(input_file_path, output_file_path):
    """
    清洗单个CSV文件：
    1. 确保是逗号分隔
    2. 删除空列或非数值列（如type列）
    3. 确保label列是0/1格式
    """
    try:
        # 读取CSV文件，使用多种编码尝试
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file_path, encoding=encoding, engine='python')
                print(f"使用编码 {encoding} 成功读取文件")
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        if df is None:
            print(f"无法读取文件 {input_file_path}")
            return False
        
        print(f"原始列: {list(df.columns)}")
        print(f"原始形状: {df.shape}")
        
        # 删除完全为空的列
        initial_cols = len(df.columns)
        df = df.dropna(axis=1, how='all')
        dropped_empty = initial_cols - len(df.columns)
        if dropped_empty > 0:
            print(f"删除了 {dropped_empty} 个全空列")
        
        # 删除完全相同的列名（重复列）
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 删除特定不需要的列（仅删除明确的空列或type列）
        columns_to_drop = []
        for col in df.columns:
            if 'type' in col.lower():  # 包含type的列
                columns_to_drop.append(col)
            elif df[col].dtype == 'object' and df[col].dropna().empty:  # 完全为空的object列
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"删除了以下列: {columns_to_drop}")
        
        # 确保label列是0/1格式
        if 'label' in df.columns:
            # 统一label值：非0值转为1，空值或无效值转为0
            def clean_label(value):
                if pd.isna(value):
                    return 0
                str_val = str(value).strip().lower()
                if str_val in ['true', 'yes', 'y', 't', '1']:
                    return 1
                elif str_val in ['false', 'no', 'n', 'f', '0']:
                    return 0
                else:
                    try:
                        num_val = float(value)
                        return 1 if num_val != 0 else 0
                    except (ValueError, TypeError):
                        return 0
            
            df['label'] = df['label'].apply(clean_label)
            print(f"label列已标准化")
        
        # 如果有bug列，也统一格式
        if 'bug' in df.columns:
            df['bug'] = df['bug'].apply(clean_label)
            print(f"bug列已标准化")
        
        # 确保所有列名都是有效的
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # 保存为标准逗号分隔的CSV，确保格式正确
        df.to_csv(
            output_file_path, 
            index=False, 
            sep=',', 
            quoting=csv.QUOTE_MINIMAL, 
            encoding='utf-8',
            lineterminator='\n'
        )
        print(f"已清洗并保存到: {output_file_path}")
        print(f"清洗后形状: {df.shape}")
        return True
        
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {str(e)}")
        return False

def clean_all_csv_in_folder(folder_path, output_folder=None):
    """
    清洗指定文件夹下的所有CSV文件
    """
    if output_folder is None:
        output_folder = folder_path
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        print(f"\n正在处理: {os.path.basename(csv_file)}")
        output_file = os.path.join(output_folder, os.path.basename(csv_file))
        clean_csv_file(csv_file, output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # 默认使用当前目录
        folder_path = input("请输入CSV文件所在的文件夹路径: ").strip()
        output_folder = input("请输入输出文件夹路径（直接回车使用相同目录）: ").strip()
        if not output_folder:
            output_folder = folder_path
    
    print(f"开始清洗 {folder_path} 下的所有CSV文件...")
    clean_all_csv_in_folder(folder_path, output_folder)
    print("CSV文件清洗完成！")