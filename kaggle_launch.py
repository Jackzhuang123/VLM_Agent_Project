"""
Kaggle Notebook Launch Script
Copy this into a Kaggle Notebook to train the VLM-VLA Agent

This script is designed to run directly in Kaggle environment.
It handles path verification, dependency installation, and training.
"""

import os
import sys


def run_command(cmd, description=""):
    """Run shell command and handle errors"""
    print(f"\n{'='*60}")
    if description:
        print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")

    result = os.system(cmd)
    if result != 0:
        print(f"❌ Command failed with exit code {result}")
        return False
    return True


def verify_kaggle_paths():
    """Verify all required paths in Kaggle environment"""
    print("\n" + "="*60)
    print("🔍 Verifying Kaggle Paths")
    print("="*60)

    paths = {
        "Input Directory": "/kaggle/input",
        "Working Directory": "/kaggle/working",
        "Dataset": "/kaggle/input/levir-cc-dateset/LEVIR-CC",
        "CLIP Model": "/kaggle/input/clip-vit-b32",
        "Qwen Model": "/kaggle/input/qwen2.5-0.5b",
    }

    all_valid = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists and "Dataset" in name or "Model" in name:
            all_valid = False

    print("="*60 + "\n")
    return all_valid


def list_kaggle_inputs():
    """List all datasets available in /kaggle/input"""
    print("\n" + "="*60)
    print("📂 Available Datasets in /kaggle/input:")
    print("="*60)

    if os.path.exists("/kaggle/input"):
        for item in os.listdir("/kaggle/input"):
            item_path = os.path.join("/kaggle/input", item)
            if os.path.isdir(item_path):
                # Count files inside
                try:
                    file_count = len(os.listdir(item_path))
                    print(f"  📁 {item}/ ({file_count} items)")
                except:
                    print(f"  📁 {item}/")

    print("="*60 + "\n")


def setup_kaggle_environment():
    """
    Main Kaggle setup and training pipeline

    This function should be called in a Kaggle Notebook cell
    """

    print("\n" + "="*70)
    print(" " * 15 + "🚀 VLM-VLA Agent - Kaggle Training")
    print("="*70 + "\n")

    # Step 1: List available inputs
    list_kaggle_inputs()

    # Step 2: Verify paths
    if not verify_kaggle_paths():
        print("⚠️  WARNING: Some required paths not found!")
        print("Please ensure all 3 datasets are added to this notebook:")
        print("  1. levir-cc-dateset")
        print("  2. clip-vit-b32")
        print("  3. qwen2.5-0.5b")
        return False

    # Step 3: Clone repository
    repo_url = "https://github.com/your-username/VLM_Agent_Project.git"
    print("\n" + "="*60)
    print("📥 Cloning Repository")
    print("="*60)
    print(f"Repository: {repo_url}")
    print("="*60 + "\n")

    # Check if already cloned
    if os.path.exists("/kaggle/working/VLM_Agent_Project"):
        print("✅ Repository already exists, skipping clone")
    else:
        print("📥 Cloning from GitHub...")
        result = os.system(f"git clone {repo_url} /kaggle/working/VLM_Agent_Project")
        if result != 0:
            print("❌ Failed to clone repository")
            return False

    # Step 4: Install dependencies
    print("\n" + "="*60)
    print("📦 Installing Dependencies")
    print("="*60)
    print("(Installing packages missing in Kaggle environment)")
    print("="*60 + "\n")

    # Only install what's not pre-installed
    packages_to_install = [
        ("datasets", "HuggingFace datasets library"),
        ("bitsandbytes", "Quantization support"),
        ("peft", "LoRA fine-tuning"),
    ]

    for package, description in packages_to_install:
        print(f"\n📦 Installing {package} ({description})...")
        result = os.system(f"pip install -q {package}")
        if result == 0:
            print(f"✅ {package} installed")
        else:
            print(f"⚠️  {package} installation may have issues (continuing anyway)")

    # Step 5: Verify project structure
    print("\n" + "="*60)
    print("🔍 Verifying Project Structure")
    print("="*60)

    project_path = "/kaggle/working/VLM_Agent_Project"
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/dataset.py",
        "src/model.py",
        "src/train.py",
        "requirements.txt",
        "README.md",
    ]

    all_present = True
    for file in required_files:
        full_path = os.path.join(project_path, file)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_present = False

    print("="*60 + "\n")

    if not all_present:
        print("❌ Some project files are missing!")
        return False

    # Step 6: Run training
    print("\n" + "="*70)
    print(" " * 20 + "🎓 Starting Training")
    print("="*70 + "\n")

    os.chdir(project_path)

    # Import and verify configuration
    sys.path.insert(0, project_path)

    try:
        from src.config import Config
        Config.print_config()
        Config.verify_paths()

        # Start training
        from src.train import main
        main()

        print("\n" + "="*70)
        print(" " * 10 + "✅ Training Completed Successfully!")
        print("="*70)
        print(f"\n📊 Results saved in: {Config.OUTPUT_DIR}")

        return True

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# QUICK START GUIDE
# ============================================================

NOTEBOOK_CELLS = """
╔════════════════════════════════════════════════════════════════╗
║           COPY THIS INTO KAGGLE NOTEBOOK CELLS                ║
╚════════════════════════════════════════════════════════════════╝

📌 CELL 1: VERIFY PATHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os

print("📂 Checking /kaggle/input structure:")
print("="*60)

# List top-level directories
for item in os.listdir("/kaggle/input"):
    print(f"  {item}/")

print("="*60)

# Verify key paths
paths_to_check = [
    ("/kaggle/input/levir-cc-dateset/LEVIR-CC", "✅ Dataset"),
    ("/kaggle/input/clip-vit-b32/config.json", "✅ CLIP model"),
    ("/kaggle/input/qwen2.5-0.5b/config.json", "✅ Qwen model"),
]

print("\\nVerifying required paths:")
print("="*60)
for path, label in paths_to_check:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"{exists} {label}: {path}")

print("="*60)


📌 CELL 2: CLONE & INSTALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Clone the repository
!git clone https://github.com/YOUR_USERNAME/VLM_Agent_Project.git
%cd VLM_Agent_Project

# Install required packages
!pip install -q datasets bitsandbytes peft


📌 CELL 3: START TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

!python -m src.train


📌 CELL 4 (OPTIONAL): MONITOR TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check saved checkpoints
import os
import json

output_dir = "/kaggle/working/output"
for item in os.listdir(output_dir):
    item_path = os.path.join(output_dir, item)
    if os.path.isdir(item_path):
        print(f"\\n📁 Checkpoint: {item}/")

        # List files
        for file in os.listdir(item_path):
            if file.endswith(".pt"):
                size_mb = os.path.getsize(os.path.join(item_path, file)) / 1e6
                print(f"  💾 {file} ({size_mb:.1f} MB)")

        # Show metrics
        metrics_path = os.path.join(item_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"  📊 Final train loss: {metrics['train']['final_loss']:.4f}")
            print(f"  📊 Final val loss: {metrics['val']['final_loss']:.4f}")

"""


if __name__ == "__main__":
    print(NOTEBOOK_CELLS)

    # Only run setup if in Kaggle environment
    if os.path.exists("/kaggle/input"):
        print("\n" + "="*70)
        print("🔍 Kaggle environment detected! Running setup...")
        print("="*70)

        success = setup_kaggle_environment()
        sys.exit(0 if success else 1)
    else:
        print("\n" + "="*70)
        print("ℹ️  This is a Kaggle launch script")
        print("="*70)
        print("\n📖 Instructions:")
        print("1. Create a new Kaggle Notebook")
        print("2. Add these 3 datasets (right panel → Input):")
        print("   • levir-cc-dateset")
        print("   • clip-vit-b32")
        print("   • qwen2.5-0.5b")
        print("3. Set GPU Accelerator to T4 x2")
        print("4. Copy the cells from above into your notebook")
        print("5. Run each cell sequentially")

