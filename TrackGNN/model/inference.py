# Model inference
# Example: dim_8_layer_1

# Import packages
import os
import torch
import numpy as np
from model import GnnClassifier
import re

checkpoint_path = '../src_files/wandb_src_files/model_checkpoint_009.pth.tar'
output_folder_path = './output_results_sw'  # software results ref
input_folder_path = '../src_files/graph_src_files/'

checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
checkpoint_state_dict = checkpoint['model']
shape = list(checkpoint_state_dict['input_network.0.weight'].shape)


hidden_dim = shape[0]
input_dim = shape[1]
num_of_layers = 1

print(input_dim, hidden_dim)

new_state_dict = {}
for key, value in checkpoint_state_dict.items():
    new_key = key

    # Update key names based on the model structure
    new_key = re.sub(r"input_network\.0", "input_network.fc", new_key)
    new_key = re.sub(r"edge_network\.network\.(\d+)", lambda match: f"edge_network.fc{int(match.group(1)) // 2 + 1}",
                     new_key)
    new_key = re.sub(r"node_network\.network\.(\d+)", lambda match: f"node_network.fc{int(match.group(1)) // 2 + 1}",
                     new_key)

    new_state_dict[new_key] = value


model = GnnClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_of_layers=num_of_layers)
model.load_state_dict(new_state_dict)

model.eval()
print("Model loaded successfully!")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

input_files = sorted([f for f in os.listdir(input_folder_path) if f.startswith('event')])[:100]


for file_name in input_files:
    file_path = os.path.join(input_folder_path, file_name)
    with np.load(file_path, allow_pickle=True) as data:
        # Load node features and edge indices
        x = torch.tensor(data['scaled_hits'], dtype=torch.float32)  # Assuming 'scaled_hits' holds node features
        layer_id = torch.tensor(data['layer_id'], dtype=torch.float32).unsqueeze(1)
        x = torch.cat([x, layer_id], dim=1)  # Concatenate layer_id as in generate_graph function

        edge_index = torch.tensor(data['edge_index'], dtype=torch.long)  # Assuming 'edge_index' holds edge indices

    with torch.no_grad():
        output = model(x, edge_index)

    # Output results
    output_file_path = os.path.join(output_folder_path, f"{file_name}_output.txt")
    np.savetxt(output_file_path, output.cpu().numpy(), fmt='%.6f')
print("All output files on software side have been successfully saved!")