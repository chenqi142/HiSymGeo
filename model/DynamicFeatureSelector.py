import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_features=4, init_index=None):
        super(GateNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features)
        )
        self.init_index = init_index
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init_index is not None:
            with torch.no_grad():
                self.fc[2].bias[self.init_index] += 1.0

    def forward(self, global_features, training=True, temperature=1.0):
        raw_weights = self.fc(global_features)

        if training:
            selection_mask = F.gumbel_softmax(raw_weights, tau=temperature, hard=True)
        else:
            selection_mask = torch.zeros_like(raw_weights)
            selected_indices = raw_weights.argmax(dim=-1)
            selection_mask.scatter_(1, selected_indices.unsqueeze(1), 1)

        return selection_mask

class DynamicFeatureSelector(nn.Module):
    def __init__(self, input_dim, num_features=4):
        super(DynamicFeatureSelector, self).__init__()
        self.gate_network = GateNetwork(input_dim, num_features, init_index=1)

    def forward(self, global_features, fused_features, training=True):
        selection_mask = self.gate_network(global_features, training)

        stacked_features = torch.stack(fused_features, dim=1)
        selected_features = (selection_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked_features).sum(dim=1)

        return  selected_features