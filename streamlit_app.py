from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import streamlit as st
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()

# Fetch API keys from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the keys are missing and prompt for manual input
if not FIRECRAWL_API_KEY or not OPENAI_API_KEY:
    st.error("❌ API keys are missing from the .env file. Please add them or input manually.")
    # Optionally, allow the user to input manually if not in .env
    with st.sidebar:
        FIRECRAWL_API_KEY = st.text_input("Firecrawl API Key", type="password", help="Enter Firecrawl API Key")
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password", help="Enter OpenAI API Key")

if not FIRECRAWL_API_KEY or not OPENAI_API_KEY:
    st.stop()  # Stop execution if API keys are missing

class PropertyData(BaseModel):
    """Schema for property data extraction"""
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")

class PropertiesResponse(BaseModel):
    """Schema for multiple properties response"""
    properties: List[PropertyData] = Field(description="List of property details")

class LocationData(BaseModel):
    """Schema for location price trends"""
    location: str
    price_per_sqft: float
    percent_increase: float
    rental_yield: float

class LocationsResponse(BaseModel):
    """Schema for multiple locations response"""
    locations: List[LocationData] = Field(description="List of location data points")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str

class PropertyFindingAgent:
    """Agent responsible for finding properties and providing recommendations"""

    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "o3-mini"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I am a real estate expert who helps find and analyze properties based on user preferences."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_properties(
        self,
        city: str,
        max_price: float,
        property_category: str = "Residential",
        property_type: str = "Flat"
    ) -> str:
        formatted_location = city.lower()
        urls = [
            f"https://www.squareyards.com/sale/property-for-sale-in-{formatted_location}/*",
            f"https://www.99acres.com/property-in-{formatted_location}-ffid/*",
            f"https://housing.com/in/buy/{formatted_location}/{formatted_location}",
        ]

        response = self.firecrawl.extract(
            urls,
            {
                "prompt": f"""
                Extract property listings for {city} where property type is {property_type} and category is {property_category}.
                Only include properties under {max_price} Crores.
                Each property must include name, address, price, description, and type.
                """,
                "schema": PropertiesResponse.model_json_schema(),
            }
        )

        if not response or "properties" not in response.get("data", {}):
            return "⚠️ No property data could be extracted. Try with a different city or parameters."

        properties = response["data"]["properties"]
        formatted = ""
        for prop in properties:
            formatted += f"""
### 🏠 {prop['Building_name']}
- 📍 **Location**: {prop['location_address']}
- 🏷️ **Type**: {prop['Property_type']}
- 💰 **Price**: {prop['Price']}
- 📝 **Description**: {prop['Description']}

---
"""
        return formatted

    def get_location_trends(self, city: str) -> str:
        """Get price trends for different localities in the city"""
        raw_response = self.firecrawl.extract([
            f"https://www.99acres.com/property-rates-and-price-trends-in-{city.lower()}-prffid/*"
        ], {
            'prompt': """Extract price trends data for ALL major localities in the city.""",
            'schema': LocationsResponse.model_json_schema(),
        })
        # You can later expand this part to format actual data
        return "Sample location trend analysis (mocked for now)."


def create_property_agent(model_id):
    if 'property_agent' not in st.session_state:
        st.session_state.property_agent = PropertyFindingAgent(
            firecrawl_api_key=FIRECRAWL_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            model_id=model_id
        )


def main():
    st.set_page_config(
        page_title="AI Real Estate Agent",
        page_icon="🏠",
        layout="wide"
    )

    st.title("Smart Property Search Agent")

    with st.sidebar:
        st.title("⚙️ Settings")
        model_id = st.selectbox(
            "🤖 Choose Model",
            options=["o3-mini", "gpt-4o"],
            help="Pick a model (use gpt-4o if o3-mini is not available)"
        )
        create_property_agent(model_id)

    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("City", placeholder="Enter city name")
        property_category = st.selectbox("Property Category", ["Residential", "Commercial"])

    with col2:
        max_price = st.number_input("Maximum Price (in Crores)", min_value=0.0)
        property_type = st.selectbox("Property Type", ["Flat", "Individual House"])

    if st.button("🔍 Start Search", use_container_width=True):
        if 'property_agent' not in st.session_state:
            st.error("⚠️ Agent not initialized. Check API keys.")
            return

        try:
            with st.spinner("🔍 Searching for properties..."):
                property_results = st.session_state.property_agent.find_properties(
                    city=city,
                    max_price=max_price,
                    property_category=property_category,
                    property_type=property_type
                )
                st.success("✅ Property search completed!")
                st.subheader("🏘️ Property Recommendations")
                st.markdown(property_results)

            with st.spinner("📊 Analyzing location trends..."):
                location_trends = st.session_state.property_agent.get_location_trends(city)
                with st.expander("📈 Location Trends Analysis"):
                    st.markdown(location_trends)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
