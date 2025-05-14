from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import tqdm

def test(test_loader, model, device="cpu"):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1 score, accuracy, precision, and recall
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }