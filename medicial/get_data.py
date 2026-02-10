import os
import pandas as pd
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import gaussian_filter


def process_mHealth_logs(raw_dir, out_dir):
    """
    Reads all mHealth_subjectX.log files from `raw_dir`,
    processes them, and saves CSVs to `out_dir`.
    """
    # Column names (following the README description)
    col_names = [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "ecg_1", "ecg_2",
        "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
        "arm_acc_x", "arm_acc_y", "arm_acc_z",
        "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
        "arm_mag_x", "arm_mag_y", "arm_mag_z",
        "activity_label"
    ]

    # Get all .log files
    log_files = glob.glob(os.path.join(raw_dir, "mHealth_subject*.log"))
    print(f"Found {len(log_files)} log files in {raw_dir}")

    for log_path in log_files:
        # Extract subject ID from filename (assuming e.g. mHealth_subject1.log)
        filename = os.path.basename(log_path)  # "mHealth_subject1.log"
        subject_id = filename.split(".")[0].replace("mHealth_subject", "")

        print(f"Processing subject {subject_id}")

        # Read the .log file into a DataFrame
        df = pd.read_csv(
            log_path,
            header=None,
            delim_whitespace=True,
            names=col_names
        )

        # Optionally, remove rows where activity_label == 0
        df = df[df['activity_label'] != 0].reset_index(drop=True)

        # Save to CSV
        out_csv_path = os.path.join(out_dir, f"mHealth_subject{subject_id}.csv")
        df.to_csv(out_csv_path, index=False)
        print(f"Saved processed data -> {out_csv_path}")


def convert_raw_to_csv():
    RAW_DIR = "/Users/muhaoguo/Documents/study/Paper_Projects/Inverse_learning/Data/MHEALTHDATASET/raw/"
    OUT_DIR = "/Users/muhaoguo/Documents/study/Paper_Projects/Inverse_learning/Data/MHEALTHDATASET/processed/"

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    process_mHealth_logs(RAW_DIR, OUT_DIR)


def get_medical_data():

    df = pd.read_csv("/Users/muhaoguo/Documents/study/Paper_Projects/Inverse_learning/Data/MHEALTHDATASET/processed/mHealth_subject1.csv")
    df = df[df["activity_label"] != 0].reset_index(drop=True)
    df = df[df["activity_label"] == 11].reset_index(drop=True)  # 5: Climbing stairs; 12: Jump front & back (20x);  11: running
    X = df[["ankle_acc_x", "ankle_acc_y"]].values
    y = df[["ecg_1", "ecg_2"]].values
    y = y[::20, :]
    X = X[::20, :]
    X = gaussian_filter(X, sigma=1)
    y = gaussian_filter(y, sigma=1)

    indices = list(range(0, len(X), 1))

    # normalization
    input_scaler = MinMaxScaler()  # Or use StandardScaler()
    target_scaler = MinMaxScaler()
    X = input_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y)

    fig, axes = plt.subplots(2, 2, figsize=(6,  6), sharex=True)

    axes[0,0].plot(indices, X[:, 0], alpha=0.75, label="ankle_acc_x")
    axes[0,0].set_title("ankle_acc_x")

    axes[0,1].plot(indices, X[:, 1], alpha=0.75, label="ankle_acc_y")
    axes[0,1].set_title("ankle_acc_y")

    axes[1,0].plot(indices, y[:, 0], alpha=0.75, label="ecg_1")
    axes[1,0].set_title("ecg_1")

    axes[1, 1].plot(indices, y[:, 1], alpha=0.75, label="ecg_2")
    axes[1, 1].set_title("ecg_2")

    plt.tight_layout()

    return X, y



if __name__ == "__main__":
    # convert_raw_to_csv()
    get_medical_data()
    plt.show()
    pass

