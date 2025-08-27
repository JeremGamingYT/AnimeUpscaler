import os
import sys
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt


console = Console()


def ensure_cuda_or_exit() -> None:
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            console.print("[red]CUDA non disponible. Installez PyTorch GPU (CUDA 12.1) et un pilote NVIDIA à jour.[/red]")
            console.print("Visitez: https://pytorch.org/get-started/locally/")
            sys.exit(1)
    except Exception:
        console.print("[red]PyTorch introuvable. Installez torch/torchvision pour CUDA 12.1.[/red]")
        console.print("Exemple: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
        sys.exit(1)


def menu() -> int:
    table = Table(title="AnimeUpscaler (LAESR)")
    table.add_column("#", justify="right")
    table.add_column("Action")
    table.add_row("1", "Télécharger un petit dataset (Wikimedia Commons)")
    table.add_row("2", "Entraîner le modèle LAESR")
    table.add_row("3", "Upscale d'images")
    table.add_row("4", "Upscale de vidéos")
    table.add_row("5", "Quitter")
    console.print(table)
    choice = IntPrompt.ask("Choisissez une option", choices=["1", "2", "3", "4", "5"], default="5")
    return int(choice)


def main() -> None:
    choice = menu()
    if choice == 1:
        from scripts.download_commons import main as dl_main
        dl_main()
    elif choice == 2:
        ensure_cuda_or_exit()
        from scripts.train import main as train_main
        train_main()
    elif choice == 3:
        ensure_cuda_or_exit()
        from scripts.infer_image import main as infer_img
        infer_img()
    elif choice == 4:
        ensure_cuda_or_exit()
        from scripts.infer_video import main as infer_vid
        infer_vid()
    else:
        console.print("Au revoir!")


if __name__ == "__main__":
    main()


