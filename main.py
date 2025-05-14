from dataset import PysMoDataset, collate_fn_sil
import torch
from sklearn.metrics import f1_score, accuracy_score
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torch
from model import GaitModel
from train import train
from test import test


def main(oversample, batch_size, result_log="results.txt"):
    transform_normalize_sil = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    semantic_data = "silhouettes"
    traits = [
            "BFI_Openness_Label", "BFI_Conscientiousness_Label", "BFI_Extraversion_Label",
            "BFI_Agreeableness_Label", "BFI_Neuroticism_Label", "RSE_Label", "BPAQ_Hostility_Label",
            "BPAQ_VerbalAggression_Label", "BPAQ_Anger_Label", "BPAQ_PhysicalAggression_Label",
            "DASS_Depression_Label", "DASS_Anxiety_Label", "DASS_Stress_Label", "GHQ_Label",
            "OFER_ChronicFatigue_Label", "OFER_AcuteFatigue_Label", "OFER_Recovery_Label"
    ]   

    for trait in traits:
        train_dataset = PysMoDataset(semantic_data=semantic_data, trait=trait, transform_sil=transform_normalize_sil, oversample=oversample, partition="train")
        test_dataset = PysMoDataset(semantic_data=semantic_data, trait=trait, transform_sil=transform_normalize_sil, oversample=oversample, partition="test")

        print("EXPERIMENT -----------------------")
        print(f"Trait: {trait}")
        print(f"Semantic data: {semantic_data}")
        print(f"Oversample: {oversample}")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_sil)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_sil)

        model = GaitModel(num_classes=len(train_dataset.unique_labels), input_size=(96, 128, 128))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train the model
        model = train(train_dataloader, model, criterion, optimizer, num_epochs=10, device=device)
        # Test the model
        metrics = test(test_dataloader, model)

        print("EXPERIMENT RESULTS -----------------------")
        print(f"Trait: {trait}")
        print(f"Semantic data: {semantic_data}")
        print(f"Oversample: {oversample}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1_score']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")

        with open(result_log, "a") as f:
            f.write(f"Trait: {trait}\n")
            f.write(f"Semantic data: {semantic_data}\n")
            f.write(f"Oversample: {oversample}\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"Test Precision: {metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {metrics['recall']:.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PysMo Dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and testing")
    parser.add_argument("--oversample", action="store_true", help="Whether to oversample the dataset")

    args = parser.parse_args()
    batch_size = args.batch_size
    oversample = args.oversample
    main(oversample, batch_size)
