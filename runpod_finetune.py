# Check GPU availability FIRST
import torch
print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ ERROR: No GPU detected!")
    print("Please enable GPU:")
    print("  1. Go to Runtime → Change runtime type")
    print("  2. Set Hardware accelerator to GPU")
    print("  3. Click Save and restart runtime")
    raise RuntimeError("GPU required for Unsloth. Please enable GPU in Colab settings.")

# Install Unsloth and dependencies (fixed for Colab)
print("\nInstalling Unsloth and dependencies...")
print("Step 1: Fixing CUDA library paths...")
import subprocess
import os

# Try to link CUDA libraries
try:
    subprocess.run(['sudo', 'ldconfig', '/usr/lib64-nvidia'], check=False, capture_output=True)
    print("  ✓ Attempted CUDA library linking")
except:
    print("  ⚠️  Could not run ldconfig (may need manual fix)")

print("\nStep 2: Installing bitsandbytes with CUDA support...")
# Install bitsandbytes properly for Colab
!pip uninstall -y bitsandbytes -q
!pip install bitsandbytes -q

print("\nStep 3: Installing triton (required for bitsandbytes)...")
!pip install triton -q

print("\nStep 4: Installing Unsloth (will install compatible dependencies)...")
# Install Unsloth first - it will install compatible versions of peft, trl, etc.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

print("\nStep 5: Verifying installation...")
# Check versions
try:
    import peft
    print(f"  ✓ PEFT version: {peft.__version__}")
except:
    print("  ⚠️  PEFT not found")

# Step 6: xformers installation (commented out - optional and takes too long to fail)
# print("\nStep 6: Installing xformers (optional, may fail but that's OK)...")
# result = subprocess.run(['pip', 'install', '--no-deps', 'xformers<0.0.27', '-q'], 
#                        capture_output=True, text=True)
# if result.returncode == 0:
#     print("  ✓ xformers installed successfully")
# else:
#     print("  ⚠️  xformers failed to build (this is OK, not required for Unsloth)")

print("\nStep 6: Verifying bitsandbytes installation...")
try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes version: {bnb.__version__}")
    # Test if CUDA is available
    if hasattr(bnb, 'cuda_available'):
        print(f"  ✓ bitsandbytes CUDA support: {bnb.cuda_available()}")
except Exception as e:
    print(f"  ⚠️  bitsandbytes verification failed: {e}")
    print("  Note: Training will still work, but 4-bit quantization may be unavailable")

print("\n✓ Installation complete!")
print("\nNote: xformers installation is skipped (optional, not required for Unsloth)")
print("Note: If you see warnings about CUDA linking, training may still work.")
print("If training fails, try restarting the runtime and running this cell again.")
