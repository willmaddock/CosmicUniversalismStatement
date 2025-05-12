import torch
from torch.utils.data import DataLoader
from CU_LossFunction import CUAlignmentLoss

def train_symbolic_model(
    model,
    dataset,
    optimizer,
    device='cuda',
    epochs=10,
    batch_size=32,
    loss_weights=(1.0, 1.0, 1.0)
):
    """
    ZTOM-aware training loop emphasizing:
    - Symbolic memory continuity
    - Recursive temporal modeling
    - CU-aligned ethical embedding
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    criterion = CUAlignmentLoss(*loss_weights)

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            symbolic_targets = batch['symbolic'].to(device)
            ethical_targets = batch['ethics'].to(device)
            recursive_labels = batch['recursive_time'].to(device)

            # Forward pass
            output = model(inputs)

            # Assume model returns:
            # - predictions
            # - symbolic_embedding
            # - time_logits
            predictions, symbolic_embedding, time_logits = output

            # Calculate composite CU-aligned loss
            loss, symbolic_loss, ethical_loss, recursive_loss = criterion(
                predictions,
                targets,
                symbolic_embedding,
                ethical_targets,
                time_logits,
                recursive_labels,
                return_breakdown=True  # Ensure the function returns sub-losses
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Optional: print batch-level info
            print(f"[Epoch {epoch+1} | Batch {batch_idx+1}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Symbolic: {symbolic_loss:.4f} | "
                  f"Ethical: {ethical_loss:.4f} | "
                  f"Recursive: {recursive_loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")

# Ensure the script runs if called directly
if __name__ == "__main__":
    from torch import nn, optim

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x):
            out = self.fc(x)
            return out, x, x  # predictions, symbolic_embedding, time_logits

    class DummyDataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            return {
                'input': torch.randn(10),
                'target': torch.tensor(1),
                'symbolic': torch.randn(10),
                'ethics': torch.randn(10),
                'recursive_time': torch.randn(10),
            }

        def __len__(self):
            return 100

    dummy_model = DummyModel()
    dummy_data = DummyDataset()
    dummy_optim = optim.Adam(dummy_model.parameters(), lr=0.001)

    print("🧠 [ZTOM-aware CU Training Loop Initialized]")
    train_symbolic_model(dummy_model, dummy_data, dummy_optim, device='cpu', epochs=3, batch_size=8)