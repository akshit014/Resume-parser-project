import re
import os
from pathlib import Path

import spacy
from spacy.cli import download as spacy_download

import fitz  # PyMuPDF
import docx2txt


class ResumeParser:
    def __init__(self):
        """Initialize the parser with spaCy model and skills list."""
        self.nlp = self._load_spacy_model("en_core_web_sm")
        self.skills = self.load_skills()

    def _load_spacy_model(self, model_name: str):
        """Load a spaCy model; if missing, download it into the current interpreter."""
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model '{model_name}'... This may take a minute.")
            spacy_download(model_name)
            return spacy.load(model_name)

    def load_skills(self):
        """Load skills from data/skills.txt (relative to this file)."""
        base_dir = Path(__file__).resolve().parent
        skills_file = base_dir / "data" / "skills.txt"

        if skills_file.exists():
            with open(skills_file, "r", encoding="utf-8") as f:
                return [line.strip().lower() for line in f if line.strip()]
        else:
            # Not fatal—just means skills extraction returns "No skills detected".
            print(f"Note: Skills file not found at: {skills_file}")
            return []

    def extract_text(self, file_path):
        """Extract text from PDF, DOCX, or TXT files."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == ".pdf":
                text = self.extract_from_pdf(file_path)
            elif file_extension == ".docx":
                text = docx2txt.process(str(file_path)) or ""
            elif file_extension == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                print(f"Unsupported file type: {file_extension}")
                return ""

            # Normalize control characters that can break NLP
            text = text.replace("\x00", " ")
            return text
        except Exception as e:
            print(f"Error extracting text from '{file_path}': {e}")
            return ""

    def extract_from_pdf(self, pdf_path: Path):
        """Extract text from PDF using PyMuPDF (fitz)."""
        text_parts = []
        try:
            # Ensure clean close even on exceptions
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    # "text" preserves layout better than "plain"
                    text_parts.append(page.get_text("text"))
        except Exception as e:
            print(f"Error reading PDF '{pdf_path}': {e}")
        return "\n".join(text_parts)

    def extract_name(self, text):
        """Extract person name using spaCy NER with simple fallbacks."""
        doc = self.nlp(text[:1000])

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()

        # Fallback: top-of-document short line
        for line in text.splitlines()[:6]:
            line = line.strip()
            if 0 < len(line) <= 60 and 1 <= len(line.split()) <= 4:
                return line

        return "Unknown"

    def extract_email(self, text):
        """Extract email address using regex."""
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        m = email_pattern.search(text)
        return m.group(0) if m else "Not found"

    def extract_phone(self, text):
        """Extract phone number using regex (handles common Indian + generic formats)."""
        patterns = [
            re.compile(r"\+91[-.\s]?\d{5}[-.\s]?\d{5}"),               # +91 98765 43210
            re.compile(r"\b0?\d{10}\b"),                               # 09876543210 or 9876543210
            re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}"),  # generic intl
        ]

        first = None
        for pat in patterns:
            m = pat.search(text)
            if m and (first is None or m.start() < first.start()):
                first = m
        return first.group(0) if first else "Not found"

    def extract_skills(self, text):
        """Find skills from the predefined skills list."""
        if not self.skills:
            return "No skills detected"

        text_lower = text.lower()
        found = []
        for skill in self.skills:
            # Word boundaries reduce partial matches (e.g., 'C' vs. 'C++' caveat remains)
            pat = re.compile(r"\b" + re.escape(skill) + r"\b")
            if pat.search(text_lower):
                found.append(skill.title())

        # Deduplicate preserving order
        seen = set()
        uniq = [s for s in found if not (s in seen or seen.add(s))]
        return ", ".join(uniq) if uniq else "No skills detected"

    def extract_education(self, text):
        """Extract education information by capturing lines after headings until the next section."""
        education_keywords = [
            "education", "academic", "qualification", "degree", "university",
            "college", "bachelor", "master", "phd", "b.tech", "m.tech", "mba",
            "bca", "mca",
        ]
        stop_words = ["experience", "work history", "projects", "skills", "certifications"]

        lines = text.splitlines()
        capture = False
        section = []

        for line in lines:
            low = line.lower()

            if not capture and any(k in low for k in education_keywords):
                capture = True

            if capture and any(w in low for w in stop_words):
                break

            if capture and line.strip():
                section.append(line.strip())
                if len(section) > 12:  # reasonable limit
                    break

        return "\n".join(section) if section else "Not found"

    def extract_experience(self, text):
        """Extract work experience section heuristically."""
        experience_keywords = [
            "experience", "work history", "employment",
            "professional experience", "work experience",
        ]
        stop_words = ["education", "skills", "projects", "certifications"]

        lines = text.splitlines()
        capture = False
        section = []

        for line in lines:
            low = line.lower()

            if not capture and any(k in low for k in experience_keywords):
                capture = True

            if capture and any(w in low for w in stop_words):
                break

            if capture and line.strip():
                section.append(line.strip())
                if len(section) > 20:
                    break

        return "\n".join(section) if section else "Not found"

    def parse(self, file_path):
        """
        Main parsing function - extracts fields from a resume file.
        Returns a dict with all extracted fields (or None if no text could be extracted).
        """
        file_path = str(file_path)
        print(f"Parsing file: {file_path}")

        raw_text = self.extract_text(file_path)
        if not raw_text:
            print("Warning: No text extracted from file")
            return None

        parsed = {
            "name": self.extract_name(raw_text),
            "email": self.extract_email(raw_text),
            "phone": self.extract_phone(raw_text),
            "skills": self.extract_skills(raw_text),
            "education": self.extract_education(raw_text),
            "experience": self.extract_experience(raw_text),
            "raw_text": raw_text,
        }

        print(f"Successfully parsed: {parsed['name']}")
        return parsed
