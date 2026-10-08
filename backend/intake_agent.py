import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables from the root .env file
load_dotenv()

# Define the structured output schema for Parvez's intake queue
class ClientIntakeSchema(BaseModel):
    client_name: str = Field(description="Name of the client if mentioned")
    budget: str = Field(description="Estimated budget or cost range mentioned")
    timeline: str = Field(description="Desired project timeline or deadline")
    spatial_requirements: list[str] = Field(description="List of rooms, spaces, or square footage requested")
    site_location: str = Field(description="Location or address of the project site")
    raw_notes_summary: str = Field(description="A clean 2-sentence summary of the overall request")

def run_intake_agent(raw_client_message: str):
    """
    Takes scattered client messages and structures them using free Gemini.
    """
    # Initialize Google's free Gemini flash model
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    
    # Bind the schema to guarantee JSON structure
    structured_llm = llm.with_structured_output(ClientIntakeSchema)
    
    prompt = f"""
    You are the Intake Agent for SiteFlow, an architecture studio platform. 
    Extract and structure the following raw client communication into clean project requirements:
    
    RAW MESSAGE:
    "{raw_client_message}"
    """
    
    result = structured_llm.invoke(prompt)
    return result

if __name__ == "__main__":
    sample_message = "Hi, this is Rajesh. I want to build a 3BHK modern villa in Pune with a budget around 1.5 crores. We need it finished by next year."
    
    print("Processing raw intake message using free Gemini model...")
    structured_output = run_intake_agent(sample_message)
    print("\n--- Structured Intake Result for Approval Queue ---")
    print(structured_output.model_dump_json(indent=2))