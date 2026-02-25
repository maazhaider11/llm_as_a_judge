import re
from typing import List, Dict, Any, Tuple

class MarkdownParser:
    """
    Parser to extract entities and triples from Markdown text.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('* *', '')
        return text.strip()

    @staticmethod
    def split_into_blocks(raw_text: str) -> List[str]:
        lines = raw_text.split("\n")
        blocks = []
        current = []

        for line in lines:
            if re.match(r'^\d+[\.\)]', line.strip()):
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return blocks

    @staticmethod
    def extract_terms_and_triples(text: str, block_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        triples = []
        terms = []

        patterns = [
            # 1) TERM means DEFINITION
            (r'"([^"]+?)"\s+means\s+(.*?)[;.]', "means"),
            # 2) TERM includes DEFINITION
            (r'"([^"]+?)"\s+includes\s+(.*?)[;.]', "includes"),
            # 3) TERM is DEFINITION
            (r'"([^"]+?)"\s+is\s+(.*?)[;.]', "is"),
            # 4) SUBJECT shall ACTION
            (r'(\b[A-Z][A-Za-z ]+?)\s+shall\s+(.*?)[;.]', "shall"),
            # 5) SUBJECT may ACTION
            (r'(\b[A-Z][A-Za-z ]+?)\s+may\s+(.*?)[;.]', "may"),
            # 6) LIABILITY
            (r'liability\s+to\s+pay\s+(\w+)\s+shall\s+be\s+of\s+(.*?)[;.]', "liability"),
            # 7) REFERENCES (section, schedule, chapter)
            (r'(section|schedule|chapter)\s+(\d+)', "refers_to"),
        ]

        for pattern, predicate in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                try:
                    subj = m.group(1).strip()
                    obj = m.group(2).strip()

                    triples.append({
                        "subject": subj,
                        "predicate": predicate,
                        "object": obj,
                        "source": block_id
                    })
                    terms.append(subj)
                except:
                    continue

        return list(set(terms)), triples

    def parse_markdown(self, text: str, document_name: str) -> Dict[str, Any]:
        """
        Parse Markdown text into blocks and triples.
        """
        blocks = self.split_into_blocks(text)
        parsed_blocks = []
        all_triples = []

        for idx, block in enumerate(blocks):
            clean = self.clean_text(block)
            block_id = f"{document_name.upper()}-BLOCK-{idx}"

            terms, local_triples = self.extract_terms_and_triples(clean, block_id)

            parsed_blocks.append({
                "id": block_id,
                "raw_text": clean,
                "terms": terms
            })
            all_triples.extend(local_triples)

        return {
            "blocks": parsed_blocks,
            "triples": all_triples
        }
