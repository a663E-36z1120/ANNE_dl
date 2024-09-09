import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef


def plot_confusion_matrix(output_path: str, logits: np.ndarray, t: np.ndarray,
                          class_names: list[str], prefix: str = '') -> None:
    y = np.argmax(logits, axis=1)
    n = y.shape[0]

    matrix = confusion_matrix(t, y)
    matrix_p = confusion_matrix(t, y, normalize='true')
    plt.matshow(matrix_p, cmap='Blues')
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    p = (np.min(matrix_p) + np.max(matrix_p)) / 2
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            color = 'white' if matrix_p[i, j] > p else 'black'
            value = f'{round(100 * matrix_p[i, j], 2)}%\n({matrix[i, j]})'
            plt.text(j, i, value, c=color, horizontalalignment='center', verticalalignment='center')

    accuracy = balanced_accuracy_score(t, y)
    precision = precision_score(t, y, average='weighted', zero_division=0)
    recall = recall_score(t, y, average='weighted', zero_division=0)
    f1 = f1_score(t, y, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(t, y)

    title = f'Sleep Stage (n = {n})'
    subtitles = ['Accuracy:', 'Precision:', 'Recall:', 'F1:', 'MCC:']
    metrics = [accuracy, precision, recall, f1, mcc]

    plt.text(-0.5, 2.75, title, weight='bold', va='top', ha='left', color='black', fontsize=10)
    plt.text(-0.5, 2.9, '\n'.join(subtitles), va='top', ha='left', color='black', fontsize=10)
    plt.text(0.1, 2.9, '\n'.join([f'{round(x, 4)}' for x in metrics]), va='top', ha='left',
             color='black', fontsize=10)

    prefix = f'{prefix}_' if len(prefix) > 0 else ''
    plt.savefig(f'{output_path}/{prefix}sleep_confusion_matrix.png', bbox_inches='tight')
    plt.close()


def plot_recordings(output_path: str, logits_all: list[np.ndarray], t_all: list[np.ndarray],
                    ids: list[str], class_names: list[str], prefix: str = '') -> None:
    cmap = plt.get_cmap('plasma')
    prefix = f'{prefix}_' if len(prefix) > 0 else ''
    path = f'{output_path}/{prefix}hypnograms.pdf'

    with PdfPages(path) as pdf:
        for id, logits, t in zip(ids, logits_all, t_all):
            y = np.argmax(logits, axis=1)
            p = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            y_display = np.cumsum(p, axis=1)
            confidence = np.max(p, axis=1)

            # Plot labels
            fig, ax = plt.subplots(3, 1, figsize=(20, 5))
            ax[0].plot(t, c='black', alpha=0.5)

            # Plot no-change points in scatter plot
            no_change = np.concatenate([[True], np.diff(y) == 0])
            ax[1].scatter(np.arange(len(confidence))[no_change], y[no_change],
                          c=((confidence[no_change] - 1 / 3) / (2 / 3)), cmap='plasma', s=1,
                          marker='s')

            # Plot transition points as lines
            for i in np.where(~no_change)[0]:
                ax[1].plot([i - 1, i], y[i - 1:i + 1], c=cmap((confidence[i] - 1 / 3) / (2 / 3)))

            # Plot softmaxes
            for i in range(y_display.shape[1]):
                if i < y_display.shape[1] - 1:
                    ax[2].plot(y_display[:, i], zorder=y_display.shape[1], c='white')
                ax[2].fill_between(range(y_display.shape[0]), 0, y_display[:, i], zorder=-i,
                                   color=cmap(i / y_display.shape[1]))

            ax[0].grid(color='black', alpha=0.1)
            ax[1].grid(color='black', alpha=0.1)

            ax[0].set_xlim([0, len(y)])
            ax[1].set_xlim([0, len(y)])
            ax[2].set_xlim([0, len(y)])
            ax[2].set_ylim([0, 1])

            ax[0].set_yticks(range(len(class_names)))
            ax[0].set_yticklabels(class_names)
            ax[1].set_yticks(range(len(class_names)))
            ax[1].set_yticklabels(class_names)

            ax[0].set_xticks([])
            ax[1].set_xticks([])
            ax[2].set_xlabel("Time")

            ax[0].set_title(f"Actual Hypnogram, {id}")
            ax[1].set_title(f"Predicted Hypnogram, {id}")
            ax[2].set_title(f"Probabilities, {id}")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def plot_training_curves(output_path: str, stats: dict) -> None:
    plt.figure()
    plt.plot(stats['train_losses'], c='black')
    plt.plot(stats['valid_losses'], c='blue')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title("Loss Curve")
    plt.grid(c='black', alpha=0.1)
    plt.savefig(f'{output_path}/loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(stats['train_accs_equal'], c='red')
    plt.plot(stats['valid_accs_equal'], c='blue')
    plt.plot(stats['train_accs_weighted'], c='darkred')
    plt.plot(stats['valid_accs_weighted'], c='darkblue')
    plt.legend(['Training Accuracy (Equal)', 'Validation Accuracy (Equal)',
                'Training Accuracy (Weighted)', 'Validation Accuracy (Weighted)'])
    plt.title("Accuracy Curve")
    plt.grid(c='black', alpha=0.1)
    plt.savefig(f'{output_path}/accuracy_curve.png')
    plt.close()

    plt.figure()
    plt.plot(stats['train_accs_stdev'], c='red')
    plt.plot(stats['valid_accs_stdev'], c='blue')
    plt.legend(['Training Accuracy Stdev', 'Validation Accuracy Stdev'])
    plt.title("Accuracy Standard Deviation Curve")
    plt.grid(c='black', alpha=0.1)
    plt.savefig(f'{output_path}/accuracy_stdev_curve.png')
    plt.close()

    plt.figure()
    plt.plot(stats['learning_rates'], c='black')
    plt.title("Learning Rates")
    plt.grid(c='black', alpha=0.1)
    plt.savefig(f'{output_path}/learning_rate.png')
    plt.close()
