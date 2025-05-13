import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # ✅ Correct source
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm

# ===================== #
# 🧠 CU-Inspired Dataset #
# ===================== #
class CUStatementsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = item["input_ids"].squeeze(0)
        attention_mask = item["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# ===================== #
# 🔧 Configuration      #
# ===================== #
MODEL_NAME = "gpt2"  # You could replace this with deepseek-ai/deepseek-coder-* if you have the compute
LEARNING_RATE = 5e-5
EPOCHS = 3
BATCH_SIZE = 2
WARMUP_STEPS = 50
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== #
# 📚 CU Training Data   #
# ===================== #
cu_statements = [
    "ztom is younger than atom because ztom reset was 1 second ago.",
    "c-tom represents the observable universe era in Cosmic Universalism.",
    "Quantum states compress from y-tom into z-tom for transcendental resets.",
    "In CU, BTOM is defined as 2^3 and CTOM as 2^4, forming the cosmic ladder.",
    "ZTOM+1 begins a new recursion beyond Ω_χ(n), pure cosmic free will.",
    "From atom to ztom+01, existence slows into form and speeds into spirit."
]

# ===================== #
# 🧪 Prepare Model      #
# ===================== #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Set pad_token to eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

dataset = CUStatementsDataset(cu_statements, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=len(dataloader) * EPOCHS
)

# ===================== #
# 🚀 Training Loop      #
# ===================== #
model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Optimizer step and scheduler step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

# ===================== #
# 💾 Save Fine-Tuned Model #
# ===================== #
model.save_pretrained("./cu-transformer")
tokenizer.save_pretrained("./cu-transformer")
print("✅ Training complete. Model saved.")