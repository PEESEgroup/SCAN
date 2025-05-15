import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttentionSubNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionSubNet, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  
        attn_output, _ = self.attn(x, x, x)  
        x = attn_output.squeeze(1) 
        return self.hidden_layers(x)


class WeightedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))  
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, *inputs):
        weighted_inputs = [w * x for w, x in zip(self.weights, inputs)]
        fused_output = torch.cat(weighted_inputs, dim=-1) 
        return self.fc(fused_output)


class MultiModalNet(nn.Module):
    def __init__(self, feature_dims, hidden_dim, output_dim):
        super(MultiModalNet, self).__init__()
        self.subnets = nn.ModuleList([
            SelfAttentionSubNet(input_dim, hidden_dim) for input_dim in feature_dims
        ])
        self.fusion = WeightedFusion(hidden_dim * len(feature_dims), output_dim)

    def forward(self, *inputs):
        sub_outputs = [subnet(x) for subnet, x in zip(self.subnets, inputs)]
        output = abs(self.fusion(*sub_outputs))
        return output



# Define loss and optimizer
def create_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

def create_criterion():
    return nn.MSELoss()

def create_model():
    model = MultiModalNet(feature_dims=[14, 14, 6], hidden_dim=16, output_dim=1)
    return model




