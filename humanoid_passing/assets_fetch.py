"""
Fetches the Robotis OP3 model from MuJoCo Menagerie.

The meshes are ~45 MB of third-party binaries, so they are downloaded on demand
and git-ignored rather than committed - the same approach the football_action_coach
package takes with its SAM 2 and ViTPose checkpoints.
"""
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotis_op3"
OP3_DIR = Path("assets/op3")


# ------------------------------------------------------------------
def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())


def fetch_op3(force: bool = False) -> int:
    """Download op3.xml, its LICENSE and every mesh it references."""
    xml = OP3_DIR / "op3.xml"
    if not xml.exists() or force:
        print(f"[Fetch] {xml}")
        _get(f"{BASE}/op3.xml", xml)
        _get(f"{BASE}/LICENSE", OP3_DIR / "LICENSE")

    meshes = sorted(set(re.findall(r'file="([^"]+)"', xml.read_text())))
    missing = [m for m in meshes if not (OP3_DIR / "assets" / m).exists() or force]
    print(f"[Fetch] {len(meshes)} meshes referenced, {len(missing)} to download")
    for i, mesh in enumerate(missing, 1):
        _get(f"{BASE}/assets/{mesh}", OP3_DIR / "assets" / mesh)
        print(f"  [{i}/{len(missing)}] {mesh}")

    print(f"[Fetch] OP3 ready in {OP3_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(fetch_op3("--force" in sys.argv))
