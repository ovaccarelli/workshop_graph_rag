"""Document handling and text extraction tools for a Pydantic AI agent."""

from pathlib import Path
from loguru import logger
from pdfminer.high_level import extract_text
from rapidocr import RapidOCR

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"


def list_my_available_documents() -> list[str]:
    """List available workshop documents.
    Returns:
        A list of document filenames available in the document directory.
    """
    return sorted(
        path.name
        for path in MY_DOCUMENTS.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )


def resolve_document_path(file_path: str) -> Path:
    """Resolve document names relative to the configured document folder.

    Names that are not absolute paths will be resolved relative to the document directory.

    Args:
        file_path: The filename or path of the document to resolve.

    Returns:
        A Path object representing the resolved absolute path to the document.
    """
    document_root = MY_DOCUMENTS.resolve()
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = document_root / path

    path = path.resolve()

    return path


def extract_text_from_pdf_file(file_path: str) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: The filename or path of the PDF to extract text from.

    Returns:
        The extracted PDF text.
    """
    path = resolve_document_path(file_path)
    logger.info(f"Extracting PDF text from {path.name}")
    return extract_text(path)


def extract_text_from_md_or_txt_file(file_path: str) -> str:
    """Extract text from a Markdown or plain text file.

    Args:
        file_path: The filename or path of the text document to extract.

    Returns:
        The text content of the document.
    """
    path = resolve_document_path(file_path)
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Only .md and .txt files are supported by this tool.")

    logger.info(f"Extracting text from {path.name}")
    return path.read_text(encoding="utf-8")


def extract_text_from_image_file(file_path: str) -> str:
    """OCR extraction for image files with uv-managed RapidOCR.

    Args:
        file_path: The filename or path of the image to extract text from.

    Returns:
        The extracted text content of the image.
    """
    path = resolve_document_path(file_path)

    logger.info(f"Extracting image text with RapidOCR from {path.name}")
    engine = RapidOCR()
    result = engine(path)
    return "\n".join(result.txts) if result else "Image text could not be extracted."
