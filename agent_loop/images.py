"""
agent_loop/images.py

Photos dragged into the terminal, kept out of the checkpoint DB.

A photo is resized and saved once to an image folder beside the DB, named by
its content hash. The message in history holds only a small reference to it,
and add_images() swaps each reference for the actual image just before the
model call. That copy is never checkpointed, so a session's size depends on
its text alone, however many photos it has.
"""

import base64
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

# Matches each image path in the input: either quoted ('/a b/c.png') or bare
# with Finder-style escaped spaces (/a\ b/c.png). Several per line is fine.
_IMAGE_EXT_RE  = r"\.(?:png|jpe?g|gif|webp|heic|heif)"
_IMAGE_PATH_RE = re.compile(
    rf"""'((?:/|~)[^']+?{_IMAGE_EXT_RE})'"""
    rf"""|"((?:/|~)[^"]+?{_IMAGE_EXT_RE})\""""
    rf"""|((?:/|~)(?:\\ |[^\s'"])+?{_IMAGE_EXT_RE})(?=$|[\s'"])""",
    re.IGNORECASE,
)

# Claude scales anything larger than this down anyway, so sending more only
# costs disk and upload time.
_MAX_EDGE = 1568
_JPEG_QUALITY = 80


class ImageStore:
    """A folder of resized photos, one file per distinct image."""

    def __init__(self, folder: "str | Path"):
        self.folder = Path(folder)

    def save(self, img_path: Path) -> str:
        """Resize img_path to a JPEG, store it, and return its reference (file name)."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "resized.jpg"
            subprocess.run(
                ["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(_JPEG_QUALITY),
                 "-Z", str(_MAX_EDGE), str(img_path), "--out", str(out)],
                check=True, capture_output=True,
            )
            data = out.read_bytes()
        ref = hashlib.sha256(data).hexdigest()[:16] + ".jpg"
        self.folder.mkdir(parents=True, exist_ok=True)
        dest = self.folder / ref
        if not dest.exists():
            dest.write_bytes(data)
        return ref

    def data_url(self, ref: str) -> "str | None":
        path = self.folder / ref
        if not path.exists():
            return None
        return "data:image/jpeg;base64," + base64.standard_b64encode(path.read_bytes()).decode()


def build_message(user_input: str, store: ImageStore) -> dict:
    """
    Build a user message. Each image path in the input (e.g. from a Finder
    drag-in) is saved to the store and becomes a reference block, which
    add_images() turns back into the image at model-call time.
    """
    matches = list(_IMAGE_PATH_RE.finditer(user_input))
    if not matches:
        return {"role": "user", "content": user_input}

    image_blocks = []
    for m in matches:
        raw_path = next(g for g in m.groups() if g).replace("\\ ", " ")
        img_path = Path(raw_path).expanduser()
        if not img_path.exists():
            print(f"  [image not found: {img_path}]")
            continue
        try:
            ref = store.save(img_path)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"  [couldn't read image {img_path.name}: {e}]")
            continue
        image_blocks.append({"type": "text", "text": f"[image: {img_path.name}]", "image_ref": ref})

    if not image_blocks:
        return {"role": "user", "content": user_input}

    text_part = _IMAGE_PATH_RE.sub("", user_input).strip()
    if not text_part:
        text_part = "Here is the image." if len(image_blocks) == 1 else "Here are the images."

    return {
        "role": "user",
        "content": [{"type": "text", "text": text_part}, *image_blocks],
    }


def add_images(messages: list, store: ImageStore) -> list:
    """Return messages with each image reference followed by the image itself.
    The originals are left untouched, so the checkpoint keeps only references."""
    out = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if not (isinstance(content, list) and any(
            isinstance(b, dict) and "image_ref" in b for b in content
        )):
            out.append(msg)
            continue
        blocks = []
        for b in content:
            if not (isinstance(b, dict) and "image_ref" in b):
                blocks.append(b)
                continue
            url = store.data_url(b["image_ref"])
            label = b["text"] if url else b["text"] + " (no longer available)"
            blocks.append({"type": "text", "text": label})
            if url:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
        out.append(msg.model_copy(update={"content": blocks}))
    return out
