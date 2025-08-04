from WT_2.models.HrMs1.model import NeuralNetwork
import torch
from WT_2.models.HrMs1.data_loader import get_data_loader
from WT_2.models.HrMs1.dataprepare import process_spectra_to_pickle

from huggingface_hub import hf_hub_download

def download_model():

    file_path = hf_hub_download(
        repo_id="liuzhenhuan123/HrMs1",
        filename="HrMs1.pth",
        local_dir="./model",
        resume_download=True
    )

    return file_path



def HrMs1Predictor(predict_msp_file, model_path=None):

    if model_path is None:
        model_path = download_model()


    process_spectra_to_pickle(predict_msp_file,
                              "./tmp.pkl", distrub=False, calculate=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 51302
    latent_dim = 512
    batch_size = 512
    neural_network = NeuralNetwork(input_dim - 2, latent_dim).to(device)
    neural_network.load_state_dict(torch.load(model_path, map_location=device))

    neural_network.eval()

    val_loader = get_data_loader("./tmp.pkl", batch_size, val=True)
    predictions = []

    with torch.no_grad():
        for data, y in val_loader:
            data = data.to(device)
            output = neural_network(data)
            predictions.append(output.cpu())
            # a = list(difference)
            # break

    # 合并结果
    predictions = torch.cat(predictions, dim=0)
    predictions_list = predictions.numpy().tolist()

    return predictions_list

