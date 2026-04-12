import tempfile
import os

try:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test")
        print(f"Success: Created {tmp.name}")
        os.unlink(tmp.name) # 清理
except Exception as e:
    print(f"Error writing to temp dir: {e}")