from typing import List, Union
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def read_pdf(path: str, return_string: bool = False) -> Union[List[Document], str]:
    """Read a PDF file and convert it to a list of documents.

    Each page of the PDF is converted to a Document object with appropriate metadata.

    Args:
        path: Path to the PDF file

    Returns:
        List of Document objects, one per page
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    converted_doc = converter.convert(path)
    md = converted_doc.document.export_to_markdown(page_break_placeholder="<!-- page break -->")
    md_remove_img = md.replace("<!-- image -->\n\n", "")

    if return_string:
        return md_remove_img

    md_split = [pg.strip() for pg in md_remove_img.split("<!-- page break -->\n\n")]
    content = []
    for i, page in enumerate(md_split):
        #TODO: Add last 128 tokens/words from previous page to the beginning of current page.
        content.append(
            Document(
                page_content=page.encode("utf-8"),
                metadata={
                    "source": path,
                    "page": i
                }
            )
        )
    
    return content