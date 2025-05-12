import torch
import torch.nn as nn

class CUAlignmentLoss(nn.Module):
    """
    Cosmic Universalism-inspired loss function.

    Encourages:
    - Temporal recursion awareness (ZTOM logic)
    - Symbolic consistency (ethically-aligned embeddings)
    - Ethical compression (aligned symbolic memory)
    """

    def __init__(self, weight_time=1.0, weight_symbol=1.0, weight_ethics=1.0):
        super(CUAlignmentLoss, self).__init__()
        self.weight_time = weight_time
        self.weight_symbol = weight_symbol
        self.weight_ethics = weight_ethics
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
            self,
            predictions,
            targets,
            symbolic_embedding,
            ethical_targets,
            time_logits,
            recursive_labels,
            return_breakdown=False
    ):
        """
        Computes composite CU-aligned loss.

        Parameters:
            predictions:       (B, C) output logits from model
            targets:           (B,) ground truth class labels
            symbolic_embedding:(B, D) symbolic representation of output
            ethical_targets:   (B, D) ethical expectation vectors
            time_logits:       (B, T) model's recursive time outputs
            recursive_labels:  (B, T) true recursive temporal structure
            return_breakdown:  if True, returns individual losses as well

        Returns:
            total_loss if return_breakdown is False
            (total_loss, loss_pred, loss_symbol, loss_time) if True
        """
        # 1. Prediction alignment (classification)
        loss_pred = self.cross_entropy(predictions, targets)

        # 2. Symbolic consistency loss
        loss_symbol = self.mse(symbolic_embedding, ethical_targets)

        # 3. Recursive time loss
        loss_time = self.mse(time_logits, recursive_labels)

        # Combine losses with weight scaling
        total_loss = (
                loss_pred +
                self.weight_symbol * loss_symbol +
                self.weight_ethics * loss_symbol +  # Symbolic ethics are embedded here
                self.weight_time * loss_time
        )

        if return_breakdown:
            return total_loss, loss_pred.item(), loss_symbol.item(), loss_time.item()
        return total_loss

# Example Usage
if __name__ == "__main__":
    # Example tensor shapes
    batch_size, num_classes, embed_size, time_steps = 32, 10, 128, 5

    # Dummy inputs
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    symbolic_embedding = torch.randn(batch_size, embed_size)
    ethical_targets = torch.randn(batch_size, embed_size)
    time_logits = torch.randn(batch_size, time_steps)
    recursive_labels = torch.randn(batch_size, time_steps)

    # Initialize loss function
    cu_loss_fn = CUAlignmentLoss()

    # Compute loss
    loss, pred_loss, symbol_loss, time_loss = cu_loss_fn(
        predictions, targets, symbolic_embedding, ethical_targets, time_logits, recursive_labels, return_breakdown=True
    )

    # Print loss breakdown
    print(f"Total Loss: {loss.item():.4f} | Pred Loss: {pred_loss:.4f} | Symbol Loss: {symbol_loss:.4f} | Time Loss: {time_loss:.4f}")
