import pandas as pd 
import time
import numpy as np 
from copy import deepcopy
from typing import Tuple
from itertools import chain 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, TensorDataset, DataLoader

from config import DATASET_ID



# # get dataset 
# def get_dataset_from_OPENML(dataset_name: str) -> pd.DataFrame:
#     dataset = openml.datasets.get_dataset(dataset_id=DATASET_ID[dataset_name])
    
#     df, _, _, _ = dataset.get_data(dataset_format='dataframe')
    
#     return df

def get_dataset_from_OPENML(dataset_name: str) -> pd.DataFrame:    
    df = pd.read_csv('openml_dataset_cache/{}_{}.csv'.format(dataset_name, DATASET_ID[dataset_name]), index_col=[0])
    
    return df



# preprocess numerical and nominial features and split training and testing
def preprocess_df(df: pd.DataFrame, numerical_cols: list, nominial_cols: list) -> Tuple[np.ndarray, np.ndarray]:
    # dropna 
    df = df.dropna()
    
    # get X and y 
    cols = df.columns 
    X = df.drop(columns=cols[-1])
    y = df[cols[-1]]
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), nominial_cols)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # if isinstance(X, np.ndarray): X_processed = preprocessor.fit_transform(X)
    # else: X_processed = preprocessor.fit_transform(X).to_array()
    
    return X_processed, y
    


def get_each_nominal_col_dim(df: pd.DataFrame, nominial_cols: list) -> dict:
    nom_col_dim_dict = {}
    for col in nominial_cols:
        nom_col_dim_dict[col] = len(df[col].unique())
        
    return nom_col_dim_dict
    

def get_feature_idx_col_name(numerical_cols: list, nominial_cols: list) -> dict:
    len_numerical_cols = len(numerical_cols)
    len_nominial_cols = len(nominial_cols)
    tot_len = len_numerical_cols + len_nominial_cols
    
    tot_cols = numerical_cols + nominial_cols
    
    idx_col_name_dict = {}
    for idx, col_name in enumerate(tot_cols):
        idx_col_name_dict[idx] = col_name
        
    return idx_col_name_dict


def get_each_feature_pos(numerical_cols: list, nominial_cols: list, nom_col_dim_dict: dict) -> dict:
    tot_cols = numerical_cols + nominial_cols
    feature_pos_dict = {}
    cnt = 0
    for col in tot_cols:
        if col not in nominial_cols:
            feature_pos_dict[col] = cnt 
            cnt += 1
        else:
            feature_dim = nom_col_dim_dict[col]
            feature_pos_dict[col] = [cnt, cnt+feature_dim]
            cnt += feature_dim
            
    return feature_pos_dict


# build dataset class 
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device=torch.device):
        super(TabularDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    
    


# build nn-based classifier
class TabularClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TabularClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Binary classification
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Skip connection with dimension matching if needed
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class TabularResNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TabularResNet, self).__init__()
        
        
        # Residual blocks
        self.layer1 = self._make_layer(input_dim, 64)
        self.layer2 = self._make_layer(64, 64)
        
        # Output layer
        self.fc = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def _make_layer(self, in_features: int, out_features: int) -> ResidualBlock:
        return ResidualBlock(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return self.softmax(x)
    


class FeatureTokenizer(nn.Module):
    def __init__(self, input_dim: int, token_dim: int):
        super(FeatureTokenizer, self).__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, token_dim),
                nn.LayerNorm(token_dim)
            ) for _ in range(input_dim)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        embedded_features = []
        for i, embedding_layer in enumerate(self.feature_embeddings):
            feature = x[:, i:i+1]  
            embedded = embedding_layer(feature)
            embedded_features.append(embedded)
        
        return torch.stack(embedded_features, dim=1)  # [batch_size, input_dim, token_dim]

class TransformerBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection and norm
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # FFN with residual and norm
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x

class FTTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, token_dim: int=64, num_heads: int=8, 
                 num_layers: int=2, dropout: float = 0.1):
        super(FTTransformer, self).__init__()
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(input_dim, token_dim)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(token_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tokenize features
        tokens = self.tokenizer(x)  # [batch_size, input_dim, token_dim]
        
        # Expand cls_token to batch_size and add to tokens
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, token_dim]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch_size, 1+input_dim, token_dim]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Use only the CLS token for prediction
        cls_token_final = tokens[:, 0]  # [batch_size, token_dim]
        
        # Pass through MLP head
        output = self.mlp_head(cls_token_final)
        
        return self.softmax(output)



# train classifier 
def train_classifier(X_train: np.ndarray, y_train: np.ndarray, output_dim: int, device: torch.device, epochs: int, batch_size: int=64, lr: float=0.001, model_type: str='mlp') -> Tuple[nn.Module, float, list]:
    # init dataset and dataloader 
    train_dataset = TabularDataset(X=X_train, y=y_train, device=device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # init classifier
    input_dim = X_train.shape[1]
    
    if model_type == 'mlp':
        model = TabularClassifier(input_dim=input_dim, output_dim=output_dim).to(device)
    elif model_type == 'resnet':
        model = TabularResNet(input_dim=input_dim, output_dim=output_dim).to(device)
    elif model_type == 'ftformer':
        model = FTTransformer(input_dim=input_dim, output_dim=output_dim).to(device)
    
    # init loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # training loop 
    running_loss_lst = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = .0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        running_loss_lst.append(running_loss/len(train_loader))
    
    end_time = time.time()
    
    return model, end_time - start_time, running_loss_lst
        
        


# evaluate classification performance 
def evaluate_classifier(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: torch.device, batch_size: int=64) -> Tuple[float, list, list]:
    # init dataset and dataloader 
    test_dataset = TabularDataset(X=X_test, y=y_test, device=device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    return accuracy, y_pred, y_true
        


# unlearn original model with our shuffling-based approach 
def train_unlearning_with_shuffle(UL_model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, device: torch.device, epochs: int, batch_size: int=64, lr: float=0.001) -> Tuple[nn.Module, list, float]: 
    # init dataset and dataloader
    train_dataset = TabularDataset(X=X_train, y=y_train, device=device)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=X_train.shape[0], shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # init loss function and optimizer
    task_loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(UL_model.parameters(), lr=lr)
    
    
    task_loss_lst = []
    start_time = time.time()
    for epoch in range(epochs):
        UL_model.train() 
        running_task_loss = 0
        for inputs, labels in train_loader:
            inputs[:, 0:1] = inputs[:, 0:1][torch.randperm(inputs.shape[0])] # shuffle unlearned feature!!!
            outputs = UL_model(inputs) # get y_hat 
            task_loss = task_loss_func(outputs, labels) # get task loss 
            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()
            
            running_task_loss += task_loss.cpu().item()

            
        print(f"Epoch [{epoch+1}/{epochs}], Running Task Loss: {running_task_loss/len(train_loader):.4f},")
        task_loss_lst.append(running_task_loss/len(train_loader))
    
    end_time = time.time()
        

    return UL_model, task_loss_lst, end_time-start_time








# baseline 1: replaced unlearned feature with uniform distribution 
def BL1_prep_data(X_train: np.ndarray, model: nn.Module) -> Tuple[nn.Module, np.ndarray]:
    # first copy origin model
    BL1_model = deepcopy(model)
    
    # replace unlearned feature with uniform distribution from -1 to 1 (for categorical feature, we can also do the same operation)
    len_X = X_train.shape[0]
    new_ul_feature = np.random.uniform(low=-1, high=1, size=len_X)
    
    BL1_X_train = deepcopy(X_train)
    BL1_X_train[:, 0] = new_ul_feature
    
    return BL1_model, BL1_X_train


# baseline 2 (NDSS 2023): Machine Unlearning of Features and Labels
def BL2_prep_data(X_train: np.ndarray, model: nn.Module):
    # first copy origin model
    BL2_model = deepcopy(model)
    
    # add little change 
    len_X = X_train.shape[0]
    delta = np.array([0.01 for _ in range(len_X)])
    
    BL2_X_train = deepcopy(X_train)
    BL2_X_train[:, 0] = BL2_X_train[:, 0] + delta 
    
    return BL2_model, BL2_X_train


def train_BL2_model(BL2_X_train: np.ndarray, y_train: np.ndarray, X_train: np.ndarray, BL2_model: nn.Module, epochs: int, device: torch.device, batch_size: int=64, lr: int=0.001):
    # generate original dataset 
    ori_train_dataset = TabularDataset(X=X_train,y=y_train, device=device)
    ori_train_loader = DataLoader(dataset=ori_train_dataset, batch_size=batch_size, shuffle=False)
    
    # generate perturbed datastet
    BL2_train_dataset = TabularDataset(X=BL2_X_train, y=y_train, device=device)
    BL2_train_loader = DataLoader(dataset=BL2_train_dataset, batch_size=batch_size, shuffle=False)
    
    # init loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BL2_model.parameters(), lr=lr)
    
    # training loop 
    running_loss_lst = []
    start_time = time.time()
    for epoch in range(epochs):
        BL2_model.train()
        running_loss = .0
        for (ori_inputs, labels), (BL2_inputs, _) in zip(ori_train_loader, BL2_train_loader):
            optimizer.zero_grad()
            ori_outputs = BL2_model(ori_inputs)
            BL2_outputs = BL2_model(BL2_inputs)
            
            ori_loss = criterion(ori_outputs, labels.to(device))
            BL2_loss = criterion(BL2_outputs, labels.to(device))
            
            # first-order update
            final_loss = -(BL2_loss - ori_loss)
            final_loss.backward()
            
            optimizer.step()
            running_loss += final_loss.cpu().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(BL2_train_loader)}")
        running_loss_lst.append(running_loss/len(BL2_train_loader))
    
    end_time = time.time()
    
    return BL2_model, end_time - start_time, running_loss_lst
    

# Baseline 3 (arXiv): Efficient Attribute Unlearning: Towards Selective Removal of Input Attributes from Feature Representations
class BL3RepDetExtractor(nn.Module):
    def __init__(self, input_dim: int):
        super(BL3RepDetExtractor, self).__init__() 
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.relu = nn.ReLU()
        
    
    def forward (self, x: torch.Tensor):
        x = self.relu(self.fc1(x))
        h = self.relu(self.fc2(x))
        
        return h


class BL3Classifier(nn.Module):
    def __init__(self, output_dim: int):
        super(BL3Classifier, self).__init__() 
        
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, h: torch.Tensor):
        h = self.fc3(h)
        
        return self.softmax(h)
    
class BL3TabularResNet(nn.Module):
    def __init__(self, output_dim: int):
        super(BL3TabularResNet, self).__init__()
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 64)
        
        # Output layer
        self.fc = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def _make_layer(self, in_features: int, out_features: int) -> ResidualBlock:
        return ResidualBlock(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return self.softmax(x)
    
    
    
class BL3FTTransformer(nn.Module):
    def __init__(self, output_dim: int, token_dim: int=64, num_heads: int=8, 
                 num_layers: int=1, dropout: float = 0.1):
        super(BL3FTTransformer, self).__init__()
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(64, token_dim)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(token_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.Linear(token_dim, output_dim)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tokenize features
        tokens = self.tokenizer(x)  # [batch_size, input_dim, token_dim]
        
        # Expand cls_token to batch_size and add to tokens
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, token_dim]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch_size, 1+input_dim, token_dim]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Use only the CLS token for prediction
        cls_token_final = tokens[:, 0]  # [batch_size, token_dim]
        
        # Pass through MLP head
        output = self.mlp_head(cls_token_final)
        
        return self.softmax(output)
    
    
    
    
class BL3DecoderMIhx(nn.Module):
    def __init__(self, output_dim: int):
        super(BL3DecoderMIhx, self).__init__()
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, h: torch.Tensor):
        h = self.relu(self.fc1(h))
        x_hat = self.sigmoid(self.fc2(h))
        
        return x_hat 
    
class BL3MIhy(nn.Module):
    def __init__(self, output_dim: int):
        super(BL3MIhy, self).__init__() 
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, h: torch.Tensor):
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        y_hat = self.softmax(self.fc3(h))
        
        return y_hat 


class BL3MIhz(nn.Module):
    def __init__(self):
        super(BL3MIhz, self).__init__() 
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, h: torch.Tensor):
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        x_hat = self.fc3(h)
        
        return x_hat




def train_BL3Backbone(X_train: np.ndarray, y_train: np.ndarray, output_dim: int, device: torch.device, epochs: int, batch_size: int=64, lr: float=0.001, model_type: str='mlp') -> Tuple[BL3RepDetExtractor, nn.Module, float, list]:
    # init dataset and dataloader 
    train_dataset = TabularDataset(X=X_train, y=y_train, device=device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # init classifier
    input_dim = X_train.shape[1]
    BL3RepDetExtractor_model = BL3RepDetExtractor(input_dim=input_dim).to(device)
    if model_type == 'mlp':
        BL3Classifier_model = BL3Classifier(output_dim=output_dim).to(device)
    elif model_type == 'resnet':
        BL3Classifier_model = BL3TabularResNet(output_dim=output_dim).to(device)
    elif model_type == 'ftformer':
        BL3Classifier_model = BL3FTTransformer(output_dim=output_dim).to(device)
        
    
    # init loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(chain(BL3RepDetExtractor_model.parameters(), BL3Classifier_model.parameters()), lr=lr)
    
    # training loop 
    running_loss_lst = []
    start_time = time.time()
    for epoch in range(epochs):
        BL3RepDetExtractor_model.train()
        BL3Classifier_model.train()
        running_loss = .0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            h = BL3RepDetExtractor_model(inputs)
            outputs = BL3Classifier_model(h)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        running_loss_lst.append(running_loss/len(train_loader))
    
    end_time = time.time()
    
    return BL3RepDetExtractor_model, BL3Classifier_model, end_time - start_time, running_loss_lst





def get_h_for_BL3(X_train: np.ndarray, device: torch.device, BL3RepDetExtractor_model: BL3RepDetExtractor, batch_size: int=64) -> np.ndarray:
    dataset = TensorDataset(torch.FloatTensor(X_train).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    h_lst = []
    BL3RepDetExtractor_model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data[0]
            h = BL3RepDetExtractor_model(data)
            h_lst.extend(h.cpu().tolist())
    
    return np.array(h_lst)


def train_BL3DecoderMIhx(h_train: np.ndarray, X_train: np.ndarray, device: torch.device, epochs: int, output_dim: int, batch_size: int=64, lr: float=0.001) -> Tuple[BL3DecoderMIhx, float]:
    dataset = TensorDataset(torch.FloatTensor(h_train).to(device), torch.FloatTensor(X_train[:, 1:]).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    BL3DecoderMIhx_model = BL3DecoderMIhx(output_dim=output_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(BL3DecoderMIhx_model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        BL3DecoderMIhx_model.train()
        running_loss = .0
        for h, x in dataloader:
            x_hat = BL3DecoderMIhx_model(h)
            loss = criterion(x_hat, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
            
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
    
    end_time = time.time()
    
    return BL3DecoderMIhx_model, end_time-start_time


def cal_BL3MIhx(BL3DecoderMIhx_model: BL3DecoderMIhx, x: torch.Tensor, h: torch.Tensor):
    BL3DecoderMIhx_model.eval()
    
    criterion = nn.MSELoss()
    x_hat = BL3DecoderMIhx_model(h)
    loss = criterion(x_hat, x)
    
    MIhx = 0.5*torch.log(2*np.pi*np.e*torch.var(x)) - loss/h.shape[0]
    
    return MIhx
    
    

def train_BL3MIhy(h_train: np.ndarray, y_train: np.ndarray, device: torch.device, epochs: int, output_dim: int, batch_size: int=64, lr: float=0.001) -> Tuple[BL3MIhy, float]:
    dataset = TensorDataset(torch.FloatTensor(h_train).to(device), torch.LongTensor(y_train).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    BL3MIhy_model = BL3MIhy(output_dim=output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BL3MIhy_model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        BL3MIhy_model.train()
        running_loss = .0
        for h, y in dataloader:
            y_hat = BL3MIhy_model(h)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
    end_time = time.time()
    
    return BL3MIhy_model, end_time-start_time

def cal_BL3MIhy(output_dim: int, BL3MIhy_model: BL3MIhy, h: torch.Tensor, y: torch.Tensor):
    BL3MIhy_model.eval()
    
    criterion = nn.CrossEntropyLoss()
    y_hat = BL3MIhy_model(h)
    loss = criterion(y_hat, y)
    
    MIhy = np.log(output_dim) - loss/h.shape[0]
    
    return MIhy

def train_BL3MIhz(h_train: np.ndarray, X_train: np.ndarray, device: torch.device, epochs: int, batch_size: int=64, lr: float=0.001) -> Tuple[BL3MIhz, float]:
    dataset = TensorDataset(torch.FloatTensor(h_train).to(device), torch.FloatTensor(X_train[:, 0:1]).to(device))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    BL3MIhz_model = BL3MIhz().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(BL3MIhz_model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        BL3MIhz_model.train()
        running_loss = .0
        for h, z in dataloader:
            z_hat = BL3MIhz_model(h)
            loss = criterion(z_hat, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
    end_time = time.time()
    
    return BL3MIhz_model, end_time-start_time


def cal_BL3MIhz(BL3MIhz_model: BL3MIhz, h: torch.Tensor, z: torch.Tensor):
    BL3MIhz_model.eval()
    
    criterion = nn.MSELoss()
    z_hat = BL3MIhz_model(h)
    loss = criterion(z_hat, z)
    
    MIhz = 0.5*torch.log(2*np.pi*np.e*torch.var(z)) - loss/h.shape[0]
    
    return MIhz
    
class BL3Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, h: np.ndarray, device: torch.device):
        super(Dataset, self).__init__()
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
        self.h = torch.FloatTensor(h).to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.h[idx]
        
        

def train_unlearning_BL3(
    BL3RepDetExtractor_model: BL3RepDetExtractor, BL3Classifier_model: nn.Module, 
    BL3DecoderMIhx_model: BL3DecoderMIhx, BL3MIhy_model: BL3MIhy, BL3MIhz_model: BL3MIhz,
    X_train: np.ndarray, y_train: np.ndarray, device: torch.device,
    BL3_lamda1: float, BL3_lamda2: float, BL3_lamda3: float,
    epochs: float, output_dim: int, batch_size: int=64, lr: float=0.001
) -> Tuple[BL3RepDetExtractor, nn.Module, float, list, list]:
    train_dataset = TabularDataset(X=X_train, y=y_train, device=device) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    task_loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(chain(BL3RepDetExtractor_model.parameters(), BL3Classifier_model.parameters()), lr=lr)
    
    BL3DecoderMIhx_model.eval()
    BL3MIhy_model.eval()
    BL3MIhz_model.eval()
    
    task_loss_lst, MI_loss_lst = [], []
    start_time = time.time()
    for epoch in range(epochs):
        running_task_loss, running_MI_loss = .0, .0
        BL3RepDetExtractor_model.eval()
        BL3Classifier_model.eval()
        for inputs, labels in train_loader:
            h = BL3RepDetExtractor_model(inputs)
            preds = BL3Classifier_model(h)
            
            task_loss = task_loss_func(preds, labels)
            
            MIhx = cal_BL3MIhx(BL3DecoderMIhx_model=BL3DecoderMIhx_model, x=inputs[:, 1:], h=h)
            MIhy = cal_BL3MIhy(output_dim=output_dim, BL3MIhy_model=BL3MIhy_model, h=h, y=labels)
            MIhz = cal_BL3MIhz(BL3MIhz_model=BL3MIhz_model, h=h, z=inputs[:, 0:1])
            MIloss = -BL3_lamda1*MIhx - BL3_lamda2*MIhy - BL3_lamda3*MIhz
            
            final_loss = task_loss + MIloss
            
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            running_task_loss += task_loss.cpu().item()
            running_MI_loss += MIloss.cpu().item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Running Task Loss: {running_task_loss/len(train_loader):.4f}")
        task_loss_lst.append(running_task_loss)
        MI_loss_lst.append(running_MI_loss)
        
    
    end_time = time.time()
    
    return BL3RepDetExtractor_model, BL3Classifier_model, end_time-start_time, task_loss_lst, MI_loss_lst


def evaluate_BL3(BL3RepDetExtractor_model: BL3RepDetExtractor, BL3Classifier_model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: torch.device, batch_size: int=64) -> Tuple[float, list, list]:
    # init dataset and dataloader 
    test_dataset = TabularDataset(X=X_test, y=y_test, device=device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

    BL3RepDetExtractor_model.eval()
    BL3Classifier_model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            h = BL3RepDetExtractor_model(inputs)
            outputs = BL3Classifier_model(h)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    return accuracy, y_pred, y_true



















