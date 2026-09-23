from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = {"PNG", "JPEG"}
TARGET_SIZE = (512, 512)


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class ProcessedImage:
    sha256: str
    format: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int


def preprocess_image(content: bytes) -> ProcessedImage:
    digest = sha256(content).hexdigest()
    try:
        with Image.open(BytesIO(content)) as source:
            source.verify()
        with Image.open(BytesIO(content)) as source:
            image_format = source.format
            if image_format not in SUPPORTED_FORMATS:
                raise InvalidImage("Only PNG and JPEG images are supported in MVP v0.1.")
            width, height = source.size
            if width < 64 or height < 64:
                raise InvalidImage("The image must be at least 64 by 64 pixels.")
            _to_model_canvas(source)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage("The uploaded file is not a valid supported image.") from exc
    return ProcessedImage(digest, image_format, width, height, *TARGET_SIZE)


def _to_model_canvas(source: Image.Image) -> Image.Image:
    grayscale = ImageOps.exif_transpose(source).convert("L")
    contained = ImageOps.contain(grayscale, TARGET_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("L", TARGET_SIZE, color=0)
    offset = ((TARGET_SIZE[0] - contained.width) // 2, (TARGET_SIZE[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    canvas.load()
    return canvas


def model_image(content: bytes) -> Image.Image:
    """Validate an upload and return the image format used by the checkpoint.

    The v0.1.0 checkpoint was trained on RGB images resized directly to 224x224;
    therefore inference intentionally uses the same representation. The padded
    512x512 image remains the product-level preprocessing summary.
    """
    preprocess_image(content)
    with Image.open(BytesIO(content)) as source:
        return ImageOps.exif_transpose(source).convert("L").convert("RGB").copy()
