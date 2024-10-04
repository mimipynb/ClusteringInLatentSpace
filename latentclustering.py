"""
    latentclustering.py
    
    Contains the model used for clustering in the latent space
"""

# TODO:
# - add a check point to store the last trained inputs' embeddings
# - add a path to save Neighbourhood

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from functools import wraps
from sentence_transformers import SentenceTransformer 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

INPUT_DIM = 768 
HIDDEN_DIM = 512
OUTPUT_DIM = 1
MAX_ITER = 10
threshold = 1.0
model_card = "all-mpnet-base-v2"

class galaxy:
    @staticmethod
    def _monitor(result: torch.tensor, data_name: str = None):
        data_name = data_name if data_name is not None else "test"
        print(f"Statistics for {data_name}:")
        print(f"Mean: {result.mean().item()}")
        print(f"Min: {result.min().item()}")
        print(f"Max: {result.max().item()}")
        print(f"Std Dev: {result.std().item()}")
        print("-" * 30)

    @staticmethod
    def monitor(func):
        @wraps(func)
        def wrapper(self, inputs: torch.tensor, hit_list: torch.tensor, kill_list: torch.tensor):
            print(f"Neighbourhood Shape input for inputs: {inputs.shape}, hit: {hit_list.shape}, kill: {kill_list.shape}")
            inputs, green, red, red_loss = func(self, inputs, hit_list, kill_list)
            output = {
                'inputs': inputs,
                'green': green,
                'red': red
            }
            # Print statistics
            for data_name, result in output.items():
                galaxy._monitor(result, data_name)
            
            return inputs, green, red, red_loss
        return wrapper

    @staticmethod
    def moonphase(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call the original function
            print("\nTesting with unseen data query . . .")
            result = func(self, *args, **kwargs)
            galaxy._monitor(result)
            return
        
        return wrapper

class LossAndFound(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super(LossAndFound, self).__init__()
        self.beacon = alpha  
        self.to(device)

    def forward(self, wait_list: torch.Tensor, hit_list: torch.Tensor, kill_list: torch.Tensor) -> torch.Tensor:
        assert wait_list.shape ==  hit_list.shape and hit_list.shape == kill_list.shape, f"Expected all inputs to be of the same shape: {wait_list.shape}, {hit_list.shape} and {kill_list.shape}"
        
        wait_list, hit_list, kill_list = wait_list.squeeze(0), hit_list.squeeze(0), kill_list.squeeze(0)
        hit_loss = torch.norm((wait_list - hit_list), p=2, dim=0)
        kill_loss = torch.norm((wait_list - kill_list), p=2, dim=0)
        final_loss = torch.max(torch.tensor(0.0, device=hit_loss.device), hit_loss - kill_loss + self.beacon)
        print(f"Final Loss: {final_loss}\t Hit loss: {hit_loss}\t Kill loss: {kill_loss}")
        return final_loss, hit_loss, kill_loss
    
class Neighbourhood(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, output_dim: int = OUTPUT_DIM, loss_fn: nn.Module = LossAndFound):
        super(Neighbourhood, self).__init__()
        self.ball = nn.Linear(input_dim, hidden_dim, device=device)
        # self.grp = nn.Linear(hidden_dim, output_dim, device=device)
        self.loss = loss_fn()
        self.opt: torch.optim = optim.Adam(self.parameters(), lr=0.001)
        self.to(device)

    @galaxy.monitor
    def forward(self, inputs: torch.tensor, hit_input: torch.tensor = None, kill_input: torch.tensor = None):
        assert inputs.size() == torch.Size([1, 768])
        if self.train:
            assert inputs.size() == hit_input.size() and inputs.size() == kill_input.size(), f"Input, target and kill input sizes are different: {inputs.size()}, {hit_input.size()}, {kill_input.size()}"
            self.opt.zero_grad()
            x = self.ball(inputs).tanh()
            pos = self.ball(hit_input).tanh()
            neg = self.ball(kill_input).tanh()
            loss, _, red_loss = self.loss(x, pos, neg)
            loss.backward()
            self.opt.step()
            return x, pos, neg, red_loss
        else:
            return self.ball(inputs).tanh()
            
class Basement(SentenceTransformer):
    def __init__(self, model_card: str = "all-mpnet-base-v2"):
        super().__init__(model_card, device=device)
        # self.encode(args, convert_to_tensor=True, device=device)

class Stadium:
    def __init__(self, model_card: str = model_card, **kwargs: dict):
        self.base = Basement(model_card)
        self.model = Neighbourhood(**kwargs)

    def update(self, *args: tuple, threshold: float = threshold, max_iter: float = MAX_ITER):
        assert len(args) == 3, f"Expected 3 str inputs for inputs, green, and red but received {len(args)}"
        self.model.train()
        inputs, green, red = self.base.encode(args, convert_to_tensor=True, device=device)
        inputs, green, red = inputs.unsqueeze(0), green.unsqueeze(0), red.unsqueeze(0)

        for ep in range(max_iter):
            new_emb, green_emb, red_emb, red_loss = self.model(inputs, green, red)
            if red_loss >= threshold:
                print(f"Training steps stopped after {ep} epochs\n")
                return
        print(f"Training steps took full {max_iter} steps\n")

    @galaxy.moonphase
    def scope(self, unseen: str):
        self.model.eval()
        embedding = self.base.encode([unseen], convert_to_tensor=True, device=device)
        with torch.no_grad():
            output= self.model.ball(embedding).tanh()
        return output 
    
if __name__ == "__main__":
    # import pandas as pd
    # initial_data = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
    input_query = "I'm surprised by the outcome."
    pos_query = "I feel happy and content"
    neg_query = "I feel sad and down."
    space = Stadium(model_card=model_card)
    space.update(input_query, pos_query, neg_query)

    unseen_query = "I am quite upset...."
    space.scope(unseen=unseen_query)