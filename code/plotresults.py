import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(output_path):
    # Load the metrics from the CSV file
    df = pd.read_csv(os.path.join(output_path, "metrics_ED_A.csv"))

    # Extract the relevant data
    # epe_mean = df["average_epe_mean"].iloc[0]
    # epe_median = df["average_epe_median"].iloc[0]
    # epe_jointwise = df[["jointwise_epe_mean_1", "jointwise_epe_mean_2", "jointwise_epe_mean_3", "jointwise_epe_mean_4", "jointwise_epe_mean_5"]].iloc[0].values
    auc = df["average_auc"].iloc[0]
    pck = eval(df["pck_curve"].iloc[0])  # Convert the string back to a list
    thresholds = eval(df["thresholds"].iloc[0])  # Convert the string back to a list

    # Plotting PCK Curve
    plt.plot(thresholds, pck, label="PCK Curve")
    plt.xlabel("Error Threshold (mm)")
    plt.ylabel("3D PCK")
    plt.title(f"PCK Curve of A on EgoDexter (AUC: {auc:.3f})")
    plt.legend()
    plt.grid()
    plt.show()

    # # Print the metrics
    # print(f"Average EPE Mean: {epe_mean}")
    # print(f"Joint-wise EPE Mean: {epe_jointwise}")
    # print(f"Average EPE Median: {epe_median}")
    # print(f"Average AUC: {auc}")


def plot_all_results(output_path):
    plt.figure(figsize=(8, 6))  # Set figure size

    # Iterate through all the expected metric files
    for letter in "ABCDEFGH":
        file_path = os.path.join(output_path, f"metrics_ED_{letter}.csv")
        
        # Check if the file exists to avoid errors
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Extract data
            auc = df["average_auc"].iloc[0]
            pck = eval(df["pck_curve"].iloc[0])  # Convert string to list
            thresholds = eval(df["thresholds"].iloc[0])  # Convert string to list

            # Plot each PCK curve with a unique label
            plt.plot(thresholds, pck, label=f"{letter} (AUC: {auc:.3f})")

    # Plot settings
    plt.xlabel("Error Threshold (mm)")
    plt.ylabel("3D PCK")
    # plt.title("PCK Curves on EgoDexter")
    plt.legend()
    plt.grid()
    plt.show()


def plot_all_results_DO(output_path):
    plt.figure(figsize=(8, 6))  # Set figure size

    # Iterate through all the expected metric files
    for letter in "ABCDEFGH":
        file_path = os.path.join(output_path, f"metrics_DO_{letter}.csv")
        
        # Check if the file exists to avoid errors
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Extract data
            auc = df["average_auc"].iloc[0]
            pck = eval(df["pck_curve"].iloc[0])  # Convert string to list
            thresholds = eval(df["thresholds"].iloc[0])  # Convert string to list

            # Plot each PCK curve with a unique label
            plt.plot(thresholds, pck, label=f"{letter} (AUC: {auc:.3f})")

    # Plot settings
    plt.xlabel("Error Threshold (mm)")
    plt.ylabel("3D PCK")
    # plt.title("PCK Curves on Dexter+Object")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_all_results("output_metrics")  # Change if the path is different
