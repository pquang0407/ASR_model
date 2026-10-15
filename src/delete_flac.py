import os
import glob

def delete_flac_files(root_dir):
    """
    Xoá toàn bộ file .flac trong root_dir và các thư mục con
    """
    flacs = glob.glob(os.path.join(root_dir, "**", "*.flac"), recursive=True)
    print(f"🔎 Tìm thấy {len(flacs)} file .flac trong {root_dir}")

    deleted = 0
    for fpath in flacs:
        try:
            os.remove(fpath)
            deleted += 1
        except Exception as e:
            print(f"❌ Không xoá được {fpath}: {e}")

    print(f"✅ Đã xoá {deleted}/{len(flacs)} file .flac trong {root_dir}")


if __name__ == "__main__":
    # Đường dẫn thư mục cần xoá flac
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    PROCESSED_DIR = os.path.join(project_root, "data", "processed")

    # Ví dụ: chỉ xoá trong train_360
    train360_root = os.path.join(PROCESSED_DIR, "train_360")

    delete_flac_files(train360_root)
