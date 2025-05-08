from WT_2.models.Samplealignment.model import SiameseNetwork
import torch
from WT_2.models.Samplealignment.dataset import SpectrumDataset
from WT_2.models.Samplealignment.data_prepare import process_spectra_to_pickle
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download



def download_model():

    file_path = hf_hub_download(
        repo_id="liuzhenhuan123/Samplealigment",
        filename="samplealigment.pth",
        local_dir="./model",
        resume_download=True
    )

    return file_path

def aligment(df, model_path=None):

    # df内容参照siamese_training_data.csv

    if model_path is None:
        model_path = download_model()

    process_spectra_to_pickle(df, "./tmp.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 51302
    hidden_dim = 16
    latent_dim = 64

    neural_network = SiameseNetwork(input_dim, hidden_dim, latent_dim).to(device)

    neural_network.load_state_dict(torch.load(model_path, map_location=device))

    neural_network.eval()

    dataset = SpectrumDataset("./tmp.pkl")
    dataloader = DataLoader(dataset, 128, shuffle=False)

    predictions = []

    with torch.no_grad():  # 不计算梯度, 节省内存和加快速度
        for left, right, left_rt, right_rt, label in dataloader:
            left, right = left.to(device), right.to(device)
            left_rt, right_rt = left_rt.to(device), right_rt.to(device)

            outputs = neural_network(left, right, left_rt, right_rt)
            outputs = torch.sigmoid(outputs)
            predictions.append(outputs.cpu())

    predictions = torch.cat(predictions, dim=0)
    predictions_list = predictions.tolist()

    return predictions_list


# import pandas as pd
# df = pd.read_csv(r"D:\work\WT2.0\peakalignment\test_siamese_training_data.csv")
# p = aligment(df)
# print(p)