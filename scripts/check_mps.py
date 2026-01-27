import torch
import time

def check_mps():
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
        
    if not torch.backends.mps.is_built():
        print("❌ MPS not built")
        return False
        
    print("✅ MPS available and built!")
    
    # Test tensor operation on MPS
    device = torch.device("mps")
    try:
        x = torch.ones(5, device=device)
        y = x * 2
        print(f"Tensor on MPS: {y}")
        print("✅ Basic MPS operation successful")
        return True
    except Exception as e:
        print(f"❌ MPS operation failed: {e}")
        return False

if __name__ == "__main__":
    check_mps()
