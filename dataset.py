# pytorch dataset for PysMo dataset
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

def collate_fn_sil(batch):
    data, labels = zip(*batch)
    max_length = 96
    padded_data = []
    for item in data:
        if item.shape[1] < max_length:
            padding = torch.zeros((1, max_length - item.shape[1], item.shape[2], item.shape[3], item.shape[4]))
            padded_item = torch.cat((item, padding), dim=1)
        else:
            padded_item = item[:, :max_length, :, :]
        padded_data.append(padded_item)
    data_tensor = torch.stack(padded_data)
    data_tensor = data_tensor.squeeze(1)
    data_tensor = data_tensor.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels

class PysMoDataset(Dataset):
    def __init__(self,semantic_data: str, partition: str, trait: str = "BFI_Extraversion_Label", transform_sil=None, oversample: bool = False, data_dir: str = "psimo_reduced"):
                
        self.data_dir = data_dir
        self.transform = transform_sil
        self.semantic_data = self.validate_semantic_data(semantic_data)
        # load files from root/semantic_data/<semantic_data>
        data_dir = os.path.join(self.data_dir, "semantic_data", self.semantic_data)

        # go though all the folders in the directory
        # The label follows the scheme: SubjectID_SequenceIndex_CameraView_RunIndex_Variation:
        # 000 → Subject ID
        # 00 → Sequence index (first sequence).
        # 0 → Camera viewpoint (0° frontal).
        # 0 → Run (repetition) index (first pass).
        # bg → Variation code (“carry bag”).
        # empty dataframe to store the data
        self.metadata_run = pd.DataFrame(columns=["ID", "sequence_index", "camera_view", "run_index", "variation"])
        for subject_folder in os.listdir(data_dir):
            # leading zeros with 3 digits
            subject_id = f"{int(subject_folder):03d}"
            for sequence_ in os.listdir(os.path.join(data_dir, subject_folder)):
                if sequence_.endswith(".json"):
                    sequence_ = sequence_.replace(".json", "")
                # get the squence index, camera view, run index and variation from the folder or file
                sequence_parts = sequence_.split("_")
                sequence_index = sequence_parts[1]
                camera_view = sequence_parts[2]
                run_index = sequence_parts[3]
                variation = sequence_parts[4]
                # add the data to the dataframe
                self.metadata_run = pd.concat([self.metadata_run, pd.DataFrame({"ID": [subject_id], "subject_folder":[subject_folder], "sequence_index": [sequence_index], "camera_view": [camera_view], "run_index": [run_index], "variation": [variation]})], ignore_index=True)

        # subjects metadata
        # root_dir/metadata_labels_v3.csv, the labels such as Low / High, etc.
        # store everything in another dataframe
        metadata_subject = pd.read_csv(os.path.join(self.data_dir, "metadata_labels_v3.csv"))
        # only take subjects 0-9
        self.metadata_subject = metadata_subject[metadata_subject["ID"].astype(int) < 10]
        # only take the columns we need (psychological traits)
        self.psychological_columns = [
            "ID", "BFI_Openness_Label", "BFI_Conscientiousness_Label", "BFI_Extraversion_Label",
            "BFI_Agreeableness_Label", "BFI_Neuroticism_Label", "RSE_Label", "BPAQ_Hostility_Label",
            "BPAQ_VerbalAggression_Label", "BPAQ_Anger_Label", "BPAQ_PhysicalAggression_Label",
            "DASS_Depression_Label", "DASS_Anxiety_Label", "DASS_Stress_Label", "GHQ_Label",
            "OFER_ChronicFatigue_Label", "OFER_AcuteFatigue_Label", "OFER_Recovery_Label"
        ]
        self.metadata_subject = self.metadata_subject[self.psychological_columns]
        
        self.trait_classification = self.validate_trait(trait)

        # subjects 0-7 for training, 8-9 for testing
        if partition == "train":
            self.metadata_run = self.metadata_run[self.metadata_run["ID"].astype(int) < 8]
        elif partition == "test":
            self.metadata_run = self.metadata_run[self.metadata_run["ID"].astype(int) >= 8]
        elif partition == "val":
            self.metadata_run = self.metadata_run[self.metadata_run["ID"].astype(int) == 8]
        else:
            raise ValueError(f"Invalid partition: {partition}. Must be 'train', 'val' or 'test'.")
        
        self.metadata_subject["ID"] = self.metadata_subject["ID"].astype(str).str.zfill(3)   
        # Add the ID column to the metadata_run dataframe
        self.metadata_run = self.metadata_run.merge(
            self.metadata_subject[["ID", self.trait_classification]], 
            on="ID", 
            how="left"
        )

        self.unique_labels = self.metadata_run[self.trait_classification].unique()
        self.idx2label = {i: label for i, label in enumerate(self.unique_labels)}
        self.label2idx = {label: i for i, label in enumerate(self.unique_labels)}

        # oversample the dataset to balance the classes
        if oversample and partition == "train":
            # oversample the dataset
            self.metadata_run = self.oversample_dataset()
        # reset the index
        self.metadata_run = self.metadata_run.reset_index(drop=True)

    # VALIDATION ---------------------------------------------------------------------------------------------------------

    def validate_semantic_data(self, semantic_data: str) -> None:
        # must be "skeleton" or "silhouette"
        if semantic_data not in ["skeletons", "silhouettes"]:
            raise ValueError(f"Invalid semantic data type: {semantic_data}. Must be 'skeleton' or 'silhouette'.") 
        return semantic_data
        
    def validate_trait(self, label: str) -> None:
        # must be "variation" or "metadata"
        if label not in self.psychological_columns:
            raise ValueError(f"Invalid label type: {label}. Must be one of {self.psychological_columns}.")
        return label

    # PROCESSING -------------------------------------------------------------------------------------------------------
    
    def oversample_dataset(self):
        class_counts = self.metadata_run[self.trait_classification].value_counts()
        max_count = class_counts.max()

        oversampled_dfs = []

        for label, count in class_counts.items():
            label_df = self.metadata_run[self.metadata_run[self.trait_classification] == label]
            n_samples = max_count - count
            if n_samples > 0:
                sampled_df = label_df.sample(n=n_samples, replace=True, random_state=42)
                oversampled_dfs.append(sampled_df)

        if oversampled_dfs:
            self.metadata_run = pd.concat([self.metadata_run] + oversampled_dfs, ignore_index=True)

        return self.metadata_run

    # DATASET -----------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        # get the subject id, sequence index, camera view, run index and variation from the dataframe
        subject_id = self.metadata_run.iloc[index]["ID"]
        subject_folder = self.metadata_run.iloc[index]["subject_folder"]
        sequence_index = self.metadata_run.iloc[index]["sequence_index"]
        camera_view = self.metadata_run.iloc[index]["camera_view"]
        run_index = self.metadata_run.iloc[index]["run_index"]
        variation = self.metadata_run.iloc[index]["variation"]


        label = self.metadata_subject[self.metadata_subject["ID"] == subject_id][self.trait_classification].values[0]
        label = self.label2idx[label]

        # load the data
        if self.semantic_data == "skeletons":
            # load the skeleton data from json file
            skeleton_file = os.path.join(self.data_dir, "semantic_data", self.semantic_data, subject_folder, f"{subject_id}_{sequence_index}_{camera_view}_{run_index}_{variation}.json")
            # list of dics, each dic is a frame with keypoints
            skeleton_data = pd.read_json(skeleton_file, orient="records")
            # convert to numpy array
            frames = []
            for frame in skeleton_data["keypoints"]:
                frame_array = np.array(frame).reshape(-1, 3)  # (17, 3)
                frames.append(frame_array)

            # Stack all frames: shape (T, 17, 3)
            data = torch.tensor(np.stack(frames), dtype=torch.float32)
        elif self.semantic_data == "silhouettes":
            # load the silhouette sequence, a series of pngs in the folder
            silhouette_folder = os.path.join(self.data_dir, "semantic_data", self.semantic_data, subject_folder, f"{subject_id}_{sequence_index}_{camera_view}_{run_index}_{variation}")
            # get all the png files in the folder
            silhouette_files = [f for f in os.listdir(silhouette_folder) if f.endswith(".png")]
            # sort the files by name
            silhouette_files.sort()
            # load the images
            silhouette_data = []
            for silhouette_file in silhouette_files:
                # load the png file
                silhouette_image = cv2.imread(os.path.join(silhouette_folder, silhouette_file), cv2.IMREAD_GRAYSCALE)
                if self.transform is not None:
                    silhouette_image = self.transform(silhouette_image)
                # convert to torch tensor
                if not isinstance(silhouette_image, torch.Tensor):
                    silhouette_image = torch.tensor(silhouette_image, dtype=torch.float32)
                else:
                    silhouette_image = silhouette_image.float()
                # append to the list
                silhouette_data.append(silhouette_image)
            # convert to numpy array
            silhouette_data = np.array(silhouette_data)
            # convert to torch tensor
            data = torch.tensor(silhouette_data, dtype=torch.float32)
            # add channel dimension in the beginning
            data = data.unsqueeze(0)  # (1, T, H, W)

        else:
            raise ValueError(f"Invalid semantic data type: {self.semantic_data}. Must be 'skeleton' or 'silhouette'.")
        
        return data, label
        
    def __len__(self):
        return len(self.metadata_run)


