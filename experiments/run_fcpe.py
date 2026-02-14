"""FCPE pitch detection on separated vocals."""
from common import load_vocals, save_results, print_summary, Frame, timer, enforce_memory_limit


def main():
    enforce_memory_limit(16)
    import torch
    from torchfcpe import spawn_bundled_infer_model

    audio = load_vocals(sr=16000)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    model = spawn_bundled_infer_model(device=device)

    # FCPE expects (batch, samples, 1)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    hop = 160  # 10ms at 16kHz
    target_len = (len(audio) // hop) + 1

    with timer() as t:
        with torch.no_grad():
            f0 = model.infer(
                audio_tensor, sr=16000,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=65, f0_max=2000,
                interp_uv=False,
                output_interp_target_length=target_len,
            )

    f0_np = f0.squeeze().cpu().numpy()
    hop_sec = hop / 16000

    # FCPE outputs 0 for unvoiced — use binary confidence
    frames = [
        Frame(i * hop_sec, float(f) if f > 0 else 0.0, 1.0 if f > 0 else 0.0)
        for i, f in enumerate(f0_np)
    ]

    save_results("fcpe", frames, t["elapsed"],
                 {"decoder_mode": "local_argmax", "threshold": 0.006})
    print_summary("FCPE", frames, t["elapsed"])


if __name__ == "__main__":
    main()
