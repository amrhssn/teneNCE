from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
from torch_geometric.data import Data


class TimeEncoder(nn.Module):
    """
    TimeEncoder class used in the GraphMIXER method for encoding time-related features.

    Encodes time step information into a fixed-size vector using a cosine function and linear transformation.
    https://github.com/CongWeilin/GraphMixer/blob/main/model.py#L32
    """

    def __init__(self, dim: int):
        """
        Initialize the TimeEncoder.

        Args:
            dim (int): Dimensionality of the time encoding.
        """
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of the TimeEncoder.
        """
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode time steps.

        Args:
            t (torch.Tensor): Time steps tensor.

        Returns:
            torch.Tensor: Encoded time step representations.
        """
        t = t.float()
        output = torch.cos(self.w(t.reshape((-1, 1)))).squeeze()
        return output


class MPNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) using Graph Convolutional Networks (GCN).

    Consists of multiple GCN layers with batch normalization and dropout.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ) -> None:
        """
        Initialize the MPNN.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
        """
        super(MPNN, self).__init__()
        self.mp1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.mp2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mp3 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Forward pass through the MPNN layers.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph edge indices.
            normalize (bool, optional): Whether to normalize the output features. Defaults to False.

        Returns:
            torch.Tensor: Output node features.
        """
        z = self.mp1(x, edge_index)
        z = self.bn1(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.1, training=self.training)

        z = self.mp2(z, edge_index)
        z = self.bn2(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.1, training=self.training)

        z = self.mp3(z, edge_index)
        z = self.bn3(z)
        if normalize:
            z = F.normalize(z, p=2., dim=-1)
        z = F.dropout(z, p=0.1, training=self.training)
        return z


class GGRU(nn.Module):
    """
    Graph Gated Recurrent Unit (GGRU) for state updates in graph sequences.

    Encodes the temporal network into the hidden state space by modeling time evolution through a
    discrete-time dynamical system, GGRU, which generates state trajectories.
    The GGRU allows the model to spread the state information across time-respecting paths.
    """

    def __init__(
            self,
            struct_embed_dim: int,
            state_dim: int
    ):
        """
        Initialize the GGRU.

        Args:
            struct_embed_dim (int): Dimension of the structural embeddings.
            state_dim (int): Dimension of the state vector.
        """
        super(GGRU, self).__init__()
        self.state_dim = state_dim

        self.Wi_reset = GCNConv(in_channels=struct_embed_dim, out_channels=state_dim, improved=True)
        self.Ws_reset = GCNConv(in_channels=state_dim, out_channels=state_dim, improved=True)

        self.Wi_update = GCNConv(in_channels=struct_embed_dim, out_channels=state_dim, improved=True)
        self.Ws_update = GCNConv(in_channels=state_dim, out_channels=state_dim, improved=True)

        self.Wi_cand = GCNConv(in_channels=struct_embed_dim, out_channels=state_dim, improved=True)
        self.Ws_cand = GCNConv(in_channels=state_dim, out_channels=state_dim, improved=True)

    def forward(
            self,
            z: torch.Tensor,
            edge_index: torch.Tensor,
            s: torch.Tensor,
            edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass to update state using GGRU.

        Args:
            z (torch.Tensor): Structural embeddings.
            edge_index (torch.Tensor): Graph edge indices.
            s (torch.Tensor): Current state vector.
            edge_weight (torch.Tensor, optional): Edge weights. Defaults to None.

        Returns:
            torch.Tensor: Updated state vector.
        """
        reset_gate = torch.sigmoid(
            self.Wi_reset(z, edge_index, edge_weight) + self.Ws_reset(s, edge_index, edge_weight))
        update_gate = torch.sigmoid(
            self.Wi_update(z, edge_index, edge_weight) + self.Ws_update(s, edge_index, edge_weight))
        s_candidate = torch.tanh(
            self.Wi_cand(z, edge_index, edge_weight) + reset_gate * self.Ws_cand(s, edge_index, edge_weight))
        s = (1 - update_gate) * s_candidate + update_gate * s
        return s


class TENENCE(nn.Module):
    """
    Temporal Network Noise Contrastive Estimation (TENENCE) model for temporal link prediction in dynamic graphs.

    The model consists of Encoder, Update, LinkPredictor, LocalPredictiveEncoder,
    GlobalPredictiveEncoder and TimeEncoder.
    The model also computes the loss values on the forward pass of the training.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ) -> None:
        """
        Initialize the TENENCE model.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
        """
        super(TENENCE, self).__init__()
        self.output_dim = output_dim
        self.encoder = GAE(encoder=MPNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim))
        self.update = GGRU(struct_embed_dim=2 * output_dim, state_dim=output_dim)
        self.time_encoder = TimeEncoder(dim=output_dim)
        self.decoder = nn.Linear(in_features=output_dim, out_features=output_dim)
        self.link_predictor = nn.Linear(in_features=output_dim, out_features=output_dim)
        self.local_predictive_encoder = nn.Sequential(
            nn.Linear(in_features=2 * output_dim, out_features=2 * output_dim),
            nn.ReLU(),
            nn.Linear(in_features=2 * output_dim, out_features=output_dim),
        )
        self.global_predictive_encoder = nn.Linear(in_features=2 * output_dim, out_features=output_dim)

    def forward(
            self,
            snapshot_sequence: List[Data],
            alpha: float = 1.0,
            beta: float = 1.0,
            normalize: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the TENENCE model to compute the loss.

        Args:
            snapshot_sequence (List[Data]): Sequence of graph snapshots.
            alpha (float, optional): Weight for the reconstruction loss. Defaults to 1.0.
            beta (float, optional): Weight for the contrastive predictive coding loss. Defaults to 1.0.
            normalize (bool, optional): Whether to normalize the output features. Defaults to False.

        Returns:
            torch.Tensor: Total loss combining prediction, reconstruction, and contrastive losses.
        """
        # Encoding the snapshot sequence
        states, state, Z_enc, Z_dec, Z_pred = self.encode_sequence(snapshot_sequence, normalize)

        # Computing losses
        prediction_loss, reconstruction_loss, cpc_loss = self.compute_losses(snapshot_sequence,
                                                                             states, Z_enc, Z_dec, Z_pred)
        loss = prediction_loss + alpha * reconstruction_loss + beta * cpc_loss
        return loss

    def encode_sequence(
            self,
            snapshot_sequence: List[Data],
            normalize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a sequence of graph snapshots.

        Args:
            snapshot_sequence (List[Data]): Sequence of graph snapshots.
            normalize (bool, optional): Whether to normalize the output features. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - states: Sequence of states at each time step.
                - state: Final state.
                - Z_enc: Encoded graph features.
                - Z_dec: Decoded graph features.
                - Z_pred: Encoded graph features for future timesteps.
        """
        num_nodes = snapshot_sequence[0].x.size(0)
        Z_enc = []
        Z_dec = []
        Z_pred = []

        state = torch.zeros(num_nodes, self.output_dim)
        last_seen = torch.zeros(num_nodes, dtype=torch.float)

        states = []
        for k, graph in enumerate(snapshot_sequence):
            x_k = graph.x.to_dense()
            edge_index_k = graph.edge_index

            # Encoding current graph
            z_enc_k = self.encoder.encode(x_k, edge_index_k, normalize=normalize)
            Z_enc.append(z_enc_k.unsqueeze(0))

            # Updating last seen embedding for state update
            src = edge_index_k[0, :].unique()
            last_seen = last_seen.index_fill(0, src, k + 1)
            last_seen_enc_k = self.time_encoder(last_seen)

            # Updating states
            z_enc_k = torch.cat([z_enc_k, last_seen_enc_k], dim=1)
            state = self.update(z_enc_k, edge_index_k, state)
            states.append(state.unsqueeze(0))

            # Reconstructing current graph
            z_dec_k = self.decoder(state)
            Z_dec.append(z_dec_k.unsqueeze(0))

            # Predicting next graph
            z_pred_k = self.link_predictor(state)
            Z_pred.append(z_pred_k.unsqueeze(0))
        states = torch.cat(states, dim=0)
        Z_enc = torch.cat(Z_enc, dim=0)
        Z_dec = torch.cat(Z_dec, dim=0)
        Z_pred = torch.cat(Z_pred, dim=0)
        return states, state, Z_enc, Z_dec, Z_pred

    def predict_next(
            self,
            snapshot_sequence: List[Data],
            normalize: bool = False
    ) -> torch.Tensor:
        """
        Predict the next graph in the sequence.

        Args:
            snapshot_sequence (List[Data]): Sequence of graph snapshots.
            normalize (bool, optional): Whether to normalize the output features. Defaults to False.

        Returns:
            torch.Tensor: Link prediction probabilities for the next graph snapshot.
        """
        states, state, Z_enc, Z_dec, Z_pred = self.encode_sequence(snapshot_sequence, normalize)
        z_pred = Z_pred[-1]
        probs = self.encoder.decoder.forward_all(z_pred, sigmoid=True)
        return probs

    def compute_losses(
            self,
            snapshot_sequence: List[Data],
            states: torch.Tensor,
            Z_enc: torch.Tensor,
            Z_dec: torch.Tensor,
            Z_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the prediction, reconstruction, and contrastive (InfoNCE) losses.

        Args:
            snapshot_sequence (List[Data]): Sequence of graph snapshots.
            states (torch.Tensor): Sequence of states at each time step.
            Z_enc (torch.Tensor): Encoded graph features.
            Z_dec (torch.Tensor): Decoded graph features.
            Z_pred (torch.Tensor): Encoded graph features for future timesteps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - prediction_loss: Loss for predicting the next graph.
                - reconstruction_loss: Loss for reconstructing the current graph.
                - cpc_loss: Contrastive predictive coding loss.
        """
        num_timesteps = len(snapshot_sequence)
        num_nodes = snapshot_sequence[0].x.size(0)
        reconstruction_loss = torch.tensor(0.0)
        cpc_loss = torch.tensor(0.0)
        prediction_loss = torch.tensor(0.0)
        ks = torch.arange(len(snapshot_sequence)).unsqueeze(0) + 1
        ks_enc = self.time_encoder(ks)
        # losses
        for k, graph in enumerate(snapshot_sequence):
            # Computing reconstruction loss at k
            z_dec_k = Z_dec[k]
            edge_index_k = snapshot_sequence[k].edge_index
            recon_loss_k = self.encoder.recon_loss(z_dec_k, edge_index_k)
            reconstruction_loss += recon_loss_k

            # Computing prediction and infoNCE losses at k
            if k < num_timesteps - 1:
                # Computing prediction loss at k
                z_pred_k = Z_pred[k]
                edge_index_next = snapshot_sequence[k + 1].edge_index
                pred_loss_k = self.encoder.recon_loss(z_pred_k, edge_index_next)
                prediction_loss += pred_loss_k

                # Computing infoNCE loss at k
                state_k = states[k]
                ks_enc_future_expanded = ks_enc[k + 1:].unsqueeze(1).repeat(1, num_nodes, 1)
                state_k_expanded = state_k.unsqueeze(0).repeat(len(ks_enc_future_expanded), 1, 1)
                global_state_k_expanded = state_k.mean(0).unsqueeze(0).repeat(len(ks_enc[k + 1:]), 1)
                z_cpc_local_future = self.local_predictive_encoder(
                    torch.cat([state_k_expanded, ks_enc_future_expanded], dim=-1))
                z_cpc_global_future = self.global_predictive_encoder(
                    torch.cat([global_state_k_expanded, ks_enc[k + 1:]], dim=-1))
                z_local_future = Z_enc[k + 1:]
                z_global_future = Z_enc[k + 1:].mean(1)

                # Computing positive scores
                # local positive scores
                scores_same_k = torch.einsum("TND, LMD -> TNM", z_cpc_local_future, z_local_future)
                pos_scores_k_local = torch.diagonal(scores_same_k, dim1=1, dim2=2)

                # global positive scores
                pos_scores_k_global = torch.diagonal(z_cpc_global_future @ z_global_future.T)

                # Computing negative scores
                # local
                # neg_scores_same_k_different_node
                same_k_not_same_nodes_mask = ~torch.eye(num_nodes, dtype=torch.bool).unsqueeze(0).repeat(
                    len(scores_same_k), 1, 1)
                neg_scores_same_k_different_node = scores_same_k[same_k_not_same_nodes_mask]

                # neg_scores_not_same_k_all_nodes
                not_same_k_mask = ~torch.eye(num_timesteps, dtype=torch.bool)[k + 1:]
                neg_scores_not_same_k_all_nodes = []
                for idx in range(ks[:, k + 1:].size(1)):
                    neg_score_kp1 = torch.einsum("ND, TMD -> TNM", z_cpc_local_future[idx], Z_enc[not_same_k_mask[idx]])
                    neg_scores_not_same_k_all_nodes.append(neg_score_kp1.unsqueeze(0))
                neg_scores_not_same_k_all_nodes = torch.cat(neg_scores_not_same_k_all_nodes, dim=0)

                # global
                # neg_scores_k_global: not_same_k
                neg_scores_k_global = (z_cpc_global_future @ Z_enc.mean(1).T)[not_same_k_mask]

                # infoNCE loss computation
                pos_scores_k = torch.cat([pos_scores_k_local.flatten(), pos_scores_k_global.flatten()], dim=0)
                neg_scores_k = torch.cat([neg_scores_same_k_different_node.flatten(),
                                          neg_scores_not_same_k_all_nodes.flatten(),
                                          neg_scores_k_global.flatten()], dim=0)
                pos_labels = torch.ones_like(pos_scores_k)
                neg_labels = torch.zeros_like(neg_scores_k)
                infoNCE_k = F.binary_cross_entropy_with_logits(
                    input=torch.cat([pos_scores_k, neg_scores_k], dim=0),
                    target=torch.cat([pos_labels, neg_labels], dim=0),
                    pos_weight=torch.tensor(len(neg_labels) / len(pos_labels))
                )
                cpc_loss += infoNCE_k
        return prediction_loss, reconstruction_loss, cpc_loss
