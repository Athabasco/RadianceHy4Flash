"""A split head can keep its unsplit source without duplicating a shard or changing the classes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    sys.exit(77)  # same optional dependency as vhead.py

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import vhead


class VHeadTest(unittest.TestCase):
    def test_head_in_own_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            shard, index = "head.safetensors", "model.safetensors.index.json"
            head = np.arange(32, dtype=np.uint16).reshape(8, 4)
            vhead.st_write(src / shard, [("lm_head.weight", head, "BF16")])
            original = json.dumps({"weight_map": {"lm_head.weight": shard}})
            (src / index).write_text(original)
            classes = src / "widths.cls"
            classes.write_bytes(b"radiance-vhead-1 n_vocab=8 widths=3,4,5,6\n\xe4\xe4")
            plain, mtp = Path(tmp) / "plain", Path(tmp) / "mtp"
            for dst, flags in [(plain, []), (mtp, ["--keep-lm-head"])]:
                subprocess.run([sys.executable, TOOLS / "vhead.py", *flags, src, dst, classes],
                               check=True)
            base = json.loads((plain / index).read_text())["weight_map"]
            draft = json.loads((mtp / index).read_text())["weight_map"]
            self.assertEqual(draft.pop("lm_head.weight"), shard)
            self.assertEqual(draft, base)
            self.assertTrue((mtp / shard).samefile(src / shard))
            self.assertFalse((mtp / shard).is_symlink())
            self.assertEqual((mtp / "vhead.safetensors").read_bytes(),
                             (plain / "vhead.safetensors").read_bytes())

            alias = Path(tmp) / "alias"
            alias.symlink_to(src, target_is_directory=True)
            attempt = subprocess.run([sys.executable, TOOLS / "vhead.py", src, alias, classes],
                                     capture_output=True)
            self.assertNotEqual(attempt.returncode, 0)
            self.assertIn(b"DST must be separate from SRC", attempt.stderr)
            self.assertEqual((src / index).read_text(), original)
            self.assertFalse((src / "vhead.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
