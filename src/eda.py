"""Basic exploratory data analysis and plots."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.common import load_dataset, REPORT_DIR

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_dataset(path) if path else load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nClass counts:\n", df["Class"].value_counts().sort_index())
    print("\nClass percentages:\n", (df["Class"].value_counts(normalize=True) * 100).round(4))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Class Distribution (0 = Legitimate, 1 = Fraud)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Amount", hue="Class", bins=50, stat="density",
                 common_norm=False, element="step", log_scale=(False, True))
    plt.title("Transaction Amount Distribution (log count axis)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "amount_distribution.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
