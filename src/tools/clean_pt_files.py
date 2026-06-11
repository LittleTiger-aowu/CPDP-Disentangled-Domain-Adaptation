import os

# ================= 配置区域 =================
# 1. 在这里修改为您要检查的文件夹绝对路径
TARGET_FOLDER = r"E:\project\WYP"

# 2. 控制开关：
#    True  = 仅预览 (列出文件，不删除) -> 【当前默认模式】
#    False = 真正执行删除操作
DRY_RUN = False


# ===========================================

def process_pt_files(root_path, is_dry_run):
    if not os.path.exists(root_path):
        print(f"❌ 错误：路径不存在 -> {root_path}")
        return

    if not os.path.isdir(root_path):
        print(f"❌ 错误：这不是一个文件夹 -> {root_path}")
        return

    mode_text = "👁️  [预览模式] " if is_dry_run else "🔥  [执行删除] "
    print(f"{mode_text}开始扫描：{root_path}\n")

    found_count = 0
    deleted_count = 0

    # 递归遍历
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.pth'):
                file_full_path = os.path.join(dirpath, filename)
                found_count += 1

                if is_dry_run:
                    # 预览模式：只打印
                    print(f"   [待删除] {file_full_path}")
                else:
                    # 执行模式：尝试删除
                    try:
                        os.remove(file_full_path)
                        print(f"   [已删除] {file_full_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"   [失败!] {file_full_path} - 原因: {e}")

    print("\n" + "=" * 40)
    if is_dry_run:
        print(f"📊 统计：共找到 {found_count} 个 .pt 文件。")
        print("💡 提示：确认列表无误后，请将代码中的 DRY_RUN 改为 False 再运行以真正删除。")
    else:
        print(f"🎉 完成！成功删除 {deleted_count} 个文件。")
        if found_count != deleted_count:
            print(f"⚠️ 注意：有 {found_count - deleted_count} 个文件因错误未删除。")
    print("=" * 40)


if __name__ == "__main__":
    # 简单的路径检查
    if TARGET_FOLDER == "请在此处填入您的文件夹路径":
        print("❌ 请先在代码第 6 行修改 TARGET_FOLDER 为真实路径！")
        exit(1)

    process_pt_files(TARGET_FOLDER, DRY_RUN)