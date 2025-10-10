import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def compare_models(results):
    df = pd.DataFrame(results).T
    print("\n📊 Model Comparison:\n", df.round(4))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind='bar', ax=ax)
    plt.title("Model Comparison - Fake News Detection")
    plt.ylabel("Score")
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=300)
    plt.show()
