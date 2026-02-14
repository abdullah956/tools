"""PDF Processing Agent."""

import json
import os
import tempfile

from envs.env_loader import EnvLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

env_loader = EnvLoader()


class ContractorDataExtractor:
    """Extract contractor data from uploaded PDF resume/CV."""

    def __init__(self):
        """Initialize the contractor data extractor."""
        self.llm = ChatOpenAI(
            api_key=env_loader.openai_api_key, model="gpt-4", temperature=0
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )

        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract contractor information from the following text. Return the information in a JSON format with the following fields. If a field's information is not found, include it with an empty string value.

            Required fields:
            - username: Preferred username or email prefix
            - email: Email address
            - first_name: First name
            - last_name: Last name
            - occupation: Current job title or occupation
            - company: Current company name
            - role: Should be set as "contractor"

            Contractor specific fields:
            - expertise: Technical expertise and skills
            - experience: Years and description of experience
            - website: Professional website or portfolio URL
            - phone: Contact phone number
            - apps_included: Any mentioned apps or tools they work with
            - language: Primary working language(s)
            - country: Country of residence/work
            - company_name: Current or most recent company name
            - type_of_services: Types of services offered
            - countries_with_office_locations: Countries where they have offices or can work
            - about: Professional summary or background
            - availability_date: Any mentioned start date (format: YYYY-MM-DD)
            - availability_time: Any mentioned preferred working hours (format: HH:MM)

            Format the response as a valid JSON object. Include ALL fields listed above, using empty strings ("") for any information not found in the text.

            Example format:
            {{
                "username": "john.doe",
                "email": "john@example.com",
                "first_name": "John",
                "last_name": "Doe",
                ...
                "availability_time": ""
            }}

            Text: {text}
            """,
        )

        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_prompt)

    def process_pdf(self, pdf_file):
        """Process the PDF file and extract contractor information."""
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_path = temp_pdf.name

        try:
            # Load and process the PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Split text into chunks
            all_text = " ".join([page.page_content for page in pages])
            chunks = self.text_splitter.split_text(all_text)

            # Initialize results with empty strings for all fields
            combined_results = {
                "username": "",
                "email": "",
                "first_name": "",
                "last_name": "",
                "occupation": "",
                "company": "",
                "role": "contractor",
                "expertise": "",
                "experience": "",
                "website": "",
                "phone": "",
                "apps_included": "",
                "language": "",
                "country": "",
                "company_name": "",
                "type_of_services": "",
                "countries_with_office_locations": "",
                "about": "",
                "availability_date": "",
                "availability_time": "",
            }

            # Process each chunk and update results
            for chunk in chunks:
                result = self.extraction_chain.run(chunk)
                try:
                    # Parse the JSON string result
                    chunk_data = json.loads(result)
                    # Only update non-empty values
                    for key, value in chunk_data.items():
                        if value and not combined_results[key]:
                            combined_results[key] = value
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue

            return combined_results

        finally:
            # Clean up temporary file
            os.unlink(temp_path)
