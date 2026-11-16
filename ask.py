import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "phase3")
import sentencepiece as spm
from phase2.model import DecoderLM
from phase3.config_ft_H import CONFIG_FT_H as cfg

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading model on {device}...")

sp = spm.SentencePieceProcessor()
sp.Load("hindi/tokenizer/hindi_bpe.model")

model = DecoderLM(cfg).to(device)
ck = torch.load("phase3/checkpoints_ft_H/best_ft.pt", map_location=device)
model.load_state_dict(ck["model"])
model.eval()
print("Model ready. Type your question in Hindi (or 'quit' to exit).\n")

while True:
    question = input("प्रश्न: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    prompt = "प्रश्न: " + question + "\nउत्तर: "
    ids = [sp.bos_id()] + sp.EncodeAsIds(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(80):
            logits, _ = model(idx[:, -cfg.context_len:])
            next_id = logits[0, -1].argmax().item()
            if next_id == sp.eos_id():
                break
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

    answer = sp.DecodeIds(idx[0, len(ids):].tolist())
    print(f"उत्तर: {answer}\n")
