def evaluate_model(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, report, cm

def draw_confusion_matrix(cm, save_plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if save_plot:
        plt.savefig(f"confusion_matrix.jpg", format="jpg")
    else:
        plt.show()

def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)
