import requests
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------- OPTIONAL CONFIG (keep None if not available) ----------
GSAS_POINT_URL = None   # e.g., "https://<your>/gsas/point?lat={lat}&lon={lon}"
BHUVAN_WMS_GFI = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"  # example base
BHUVAN_LAYER = None     # e.g., "nbss_soil:india_soil_type"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_TIMEOUT = 15
# ---------------------------------------------------------------

def in_india(lat: float, lon: float) -> bool:
    """Check if coordinates are within India's geographical boundaries"""
    return (6.0 <= lat <= 37.2) and (68.0 <= lon <= 97.5)

class SoilDataAPI:
    """
    Enhanced Soil Data API with comprehensive soil type prediction and location info.
    
    Features:
    - Predicts one of 10 soil types: Sandy, Silty, Laterite, Alkaline, Black, Clayey, Saline, Loamy, Red, Peaty
    - Uses multiple data sources: GSAS, Bhuvan WMS, SoilGrids
    - Includes accurate location information via reverse geocoding
    - Caching for improved performance
    - Robust fallback mechanisms
    """
    
    # Valid soil types that can be returned
    VALID_SOIL_TYPES = {
        "Sandy", "Silty", "Laterite", "Alkaline", "Black", 
        "Clayey", "Saline", "Loamy", "Red", "Peaty"
    }
    
    def __init__(self):
        self.logger = logger
        self.cache = {}
        self.logger.info("SoilDataAPI initialized with enhanced features")

    def get_soil_data_by_location(self, latitude: float, longitude: float) -> Dict:
        """
        Main function to get comprehensive soil data for a location.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            Dict: Complete soil data including type, properties, confidence, sources, and location info
        """
        cache_key = f"{latitude:.5f},{longitude:.5f}"
        if cache_key in self.cache:
            self.logger.info(f"Returning cached soil data for {cache_key}")
            return self.cache[cache_key]

        # Initialize response structure
        response = {
            "location": {
                "latitude": latitude, 
                "longitude": longitude, 
                "timestamp": datetime.now().isoformat()
            },
            "soil_properties": {},
            "soil_type": "Loamy",  # Default fallback
            "confidence": 0.25,
            "sources": [],
            "success": False,
            "location_info": {}
        }

        try:
            # Get location information first
            response["location_info"] = self.get_location_info(latitude, longitude)
            
            # Get soil properties from SoilGrids
            soil_properties = self._get_soilgrids_data(latitude, longitude)
            
            if soil_properties:
                response["soil_properties"] = soil_properties
                response["sources"].append("SoilGrids")
                self.logger.info(f"SoilGrids data retrieved for {latitude}, {longitude}")
            else:
                # Fallback to mock data
                soil_properties = self._get_mock_soil_data(latitude, longitude)
                if soil_properties:
                    response["soil_properties"] = soil_properties
                    response["sources"].append("Mock")
                    self.logger.warning(f"Using mock data for {latitude}, {longitude}")

            # Determine soil type using multiple sources and priority logic
            soil_type, confidence, sources = self._determine_soil_type(
                latitude, longitude, soil_properties
            )
            
            response["soil_type"] = soil_type
            response["confidence"] = confidence
            
            # Add sources if not already present
            for source in sources:
                if source not in response["sources"]:
                    response["sources"].append(source)
            
            response["success"] = True
            
            # Cache the result
            self.cache[cache_key] = response
            
            self.logger.info(
                f"Soil analysis complete for {latitude}, {longitude}: "
                f"Type={soil_type}, Confidence={confidence:.2f}, Sources={response['sources']}"
            )
            
            return response

        except Exception as e:
            self.logger.error(f"Error getting soil data for {latitude}, {longitude}: {e}")
            # Return basic response with location info if available
            response["location_info"] = self.get_location_info(latitude, longitude)
            return response

    def _determine_soil_type(self, lat: float, lon: float, soil_properties: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """
        Determine soil type using priority-based logic with multiple data sources.
        
        Priority order:
        1. GSAS API for Saline/Alkaline soils
        2. Bhuvan WMS for Red, Black, Laterite, Peaty soils (India only)
        3. SoilGrids texture analysis for Clayey, Sandy, Silty, Loamy
        4. Fallback to Loamy
        """
        sources = []
        
        # Priority 1: Check for Saline/Alkaline soils using GSAS
        salt_alk_result = self._get_salt_alk_class(lat, lon, soil_properties)
        if salt_alk_result[0]:  # If a classification was found
            soil_type, confidence, source = salt_alk_result
            if source:
                sources.append(source)
            self.logger.info(f"Salt/Alkaline classification: {soil_type} (confidence: {confidence})")
            return soil_type, confidence, sources

        # Priority 2: Check Bhuvan for specific soil types (India only)
        if in_india(lat, lon):
            bhuvan_category = self._get_bhuvan_category(lat, lon)
            if bhuvan_category in {"Red", "Black", "Laterite", "Peaty"}:
                sources.append("Bhuvan")
                self.logger.info(f"Bhuvan classification: {bhuvan_category}")
                return bhuvan_category, 0.85, sources

        # Priority 3: Texture-based classification from SoilGrids
        if soil_properties:
            texture_type, texture_confidence = self._texture_family_from_soilgrids(soil_properties)
            self.logger.info(f"Texture classification: {texture_type} (confidence: {texture_confidence})")
            return texture_type, texture_confidence, sources

        # Priority 4: Fallback to Loamy with low confidence
        self.logger.warning(f"Using fallback classification for {lat}, {lon}")
        return "Loamy", 0.25, sources

    def _get_soilgrids_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get soil properties from SoilGrids API with batch and individual fallback support.
        
        Returns:
            Optional[Dict]: Soil properties (clay, sand, silt, phh2o) as percentages/values
        """
        try:
            # First try batch request for all properties
            batch_result = self._try_soilgrids_batch(lat, lon)
            if batch_result:
                return batch_result

            # If batch fails, try individual property requests
            self.logger.warning(f"Batch SoilGrids request failed for {lat}, {lon}, trying individual requests")
            individual_result = self._get_soilgrids_individual(lat, lon)
            if individual_result:
                return individual_result

            # Both methods failed
            self.logger.error(f"All SoilGrids requests failed for {lat}, {lon}")
            return None

        except Exception as e:
            self.logger.error(f"SoilGrids error for {lat}, {lon}: {e}")
            return None

    def _try_soilgrids_batch(self, lat: float, lon: float) -> Optional[Dict]:
        """Try to get all soil properties in a single batch request"""
        try:
            params = {
                "lon": lon,
                "lat": lat,
                "property": "clay,sand,silt,phh2o",
                "depth": "0-5cm",
                "value": "mean",
            }
            
            response = requests.get(SOILGRIDS_URL, params=params, timeout=SOILGRIDS_TIMEOUT)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            props = {}
            
            for prop in data.get("properties", []):
                name = prop.get("name")
                if not prop.get("depths"):
                    continue
                    
                surf = prop["depths"][0]
                val = surf.get("values", {}).get("mean")
                
                if val is None:
                    continue
                
                # Convert g/kg to percentage for texture properties
                if name in {"clay", "sand", "silt"}:
                    props[name] = float(val) / 10.0
                else:
                    props[name] = float(val)

            # Normalize texture percentages to sum to 100%
            if all(k in props for k in ("clay", "sand", "silt")):
                total = props["clay"] + props["sand"] + props["silt"]
                if total > 0:
                    props["clay"] = (props["clay"] / total) * 100.0
                    props["sand"] = (props["sand"] / total) * 100.0
                    props["silt"] = (props["silt"] / total) * 100.0

            return props if props else None
            
        except Exception as e:
            self.logger.warning(f"Batch SoilGrids request failed: {e}")
            return None

    def _get_soilgrids_individual(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Try individual property requests as fallback.
        
        Returns:
            Optional[Dict]: Individual soil properties
        """
        try:
            props = {}
            
            for prop_name in ["clay", "sand", "silt", "phh2o"]:
                try:
                    params = {
                        "lon": lon,
                        "lat": lat,
                        "property": prop_name,
                        "depth": "0-5cm",
                        "value": "mean",
                    }
                    
                    response = requests.get(SOILGRIDS_URL, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("properties") and data["properties"][0].get("depths"):
                            val = data["properties"][0]["depths"][0].get("values", {}).get("mean")
                            if val is not None:
                                # Convert units appropriately
                                if prop_name in {"clay", "sand", "silt"}:
                                    props[prop_name] = float(val) / 10.0  # g/kg to %
                                else:
                                    props[prop_name] = float(val)
                                self.logger.debug(f"Successfully retrieved {prop_name}: {props[prop_name]}")
                            else:
                                self.logger.warning(f"No value found for {prop_name}")
                        else:
                            self.logger.warning(f"Invalid data structure for {prop_name}")
                    else:
                        self.logger.warning(f"SoilGrids request failed for {prop_name}: Status {response.status_code}")
                                    
                except Exception as e:
                    self.logger.warning(f"Individual request failed for {prop_name}: {e}")
                    continue
            
            # Normalize texture percentages if we have all three
            if all(k in props for k in ("clay", "sand", "silt")):
                total = props["clay"] + props["sand"] + props["silt"]
                if total > 0:
                    props["clay"] = (props["clay"] / total) * 100.0
                    props["sand"] = (props["sand"] / total) * 100.0
                    props["silt"] = (props["silt"] / total) * 100.0
            
            return props if props else None
            
        except Exception as e:
            self.logger.error(f"Individual SoilGrids requests failed: {e}")
            return None

    def _get_mock_soil_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Provide realistic mock soil data based on geographical location.
        
        Returns:
            Optional[Dict]: Mock soil properties based on regional characteristics
        """
        try:
            if in_india(lat, lon):
                # India-specific regional soil characteristics
                if lat > 30:  # Northern India (Punjab, Haryana, etc.)
                    return {
                        "clay": 25.0, "sand": 45.0, "silt": 30.0, "phh2o": 7.5
                    }
                elif lat < 15:  # Southern India (Tamil Nadu, Kerala, etc.)
                    return {
                        "clay": 35.0, "sand": 40.0, "silt": 25.0, "phh2o": 6.8
                    }
                elif 20 <= lat <= 25:  # Central India (Maharashtra, MP)
                    return {
                        "clay": 40.0, "sand": 35.0, "silt": 25.0, "phh2o": 7.2
                    }
                else:  # Other Indian regions
                    return {
                        "clay": 30.0, "sand": 42.0, "silt": 28.0, "phh2o": 7.0
                    }
            else:
                # Global regions with different characteristics
                if -30 <= lat <= 30:  # Tropical regions
                    return {
                        "clay": 28.0, "sand": 52.0, "silt": 20.0, "phh2o": 6.5
                    }
                elif lat > 50 or lat < -50:  # Temperate/cold regions
                    return {
                        "clay": 22.0, "sand": 48.0, "silt": 30.0, "phh2o": 6.8
                    }
                else:  # Mid-latitude regions
                    return {
                        "clay": 25.0, "sand": 50.0, "silt": 25.0, "phh2o": 7.0
                    }
                    
        except Exception as e:
            self.logger.error(f"Error generating mock data: {e}")
            # Ultimate fallback
            return {
                "clay": 25.0, "sand": 50.0, "silt": 25.0, "phh2o": 7.0
            }

    def _get_salt_alk_class(self, lat: float, lon: float, soil_properties: Optional[Dict]) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Determine if soil is Saline or Alkaline using GSAS API and pH heuristics.
        
        Returns:
            Tuple[Optional[str], float, Optional[str]]: (soil_type, confidence, source)
        """
        # Try GSAS API if configured
        if GSAS_POINT_URL:
            try:
                url = GSAS_POINT_URL.format(lat=lat, lon=lon)
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                classification = str(data.get("class", "")).lower()
                
                if classification in {"saline", "saline-sodic"}:
                    self.logger.info(f"GSAS detected saline soil at {lat}, {lon}")
                    return "Saline", 0.9, "GSAS"
                elif classification in {"sodic", "alkaline"}:
                    self.logger.info(f"GSAS detected alkaline soil at {lat}, {lon}")
                    return "Alkaline", 0.9, "GSAS"
                    
            except Exception as e:
                self.logger.warning(f"GSAS API lookup failed for {lat}, {lon}: {e}")

        # Fallback to pH-based heuristics
        if soil_properties and "phh2o" in soil_properties:
            ph = soil_properties["phh2o"]
            
            if ph >= 8.5:
                self.logger.info(f"pH-based alkaline classification: pH={ph}")
                return "Alkaline", 0.65, "pH Heuristic"
            elif ph >= 8.3:
                self.logger.info(f"pH-based weak alkaline classification: pH={ph}")
                return "Alkaline", 0.55, "pH Heuristic"

        return None, 0.0, None

    def _get_bhuvan_category(self, lat: float, lon: float) -> Optional[str]:
        """
        Get soil category from Bhuvan WMS service for Indian coordinates.
        
        Returns:
            Optional[str]: One of Red, Black, Laterite, Peaty if detected
        """
        try:
            if not (BHUVAN_WMS_GFI and BHUVAN_LAYER):
                self.logger.debug("Bhuvan WMS not configured")
                return None
            
            # Create small bounding box around the point
            delta = 0.0005
            bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
            
            params = {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetFeatureInfo",
                "LAYERS": BHUVAN_LAYER,
                "QUERY_LAYERS": BHUVAN_LAYER,
                "CRS": "EPSG:4326",
                "INFO_FORMAT": "application/json",
                "I": "1",
                "J": "1",
                "WIDTH": "3",
                "HEIGHT": "3",
                "BBOX": bbox,
            }
            
            response = requests.get(BHUVAN_WMS_GFI, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for feature in data.get("features", []):
                properties = feature.get("properties", {})
                
                # Check various possible property names for soil type
                for key in ("SOILTYPE", "soil_type", "SOIL", "class", "TYPE", "name"):
                    value = properties.get(key)
                    if not value:
                        continue
                    
                    value_lower = str(value).lower()
                    
                    # Map to our standard soil types
                    if "red" in value_lower:
                        return "Red"
                    elif "black" in value_lower or "vertisol" in value_lower:
                        return "Black"
                    elif "laterite" in value_lower or "lateritic" in value_lower:
                        return "Laterite"
                    elif "peat" in value_lower or "histic" in value_lower or "muck" in value_lower:
                        return "Peaty"
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Bhuvan WMS GetFeatureInfo failed for {lat}, {lon}: {e}")
            return None

    def _texture_family_from_soilgrids(self, soil_properties: Optional[Dict]) -> Tuple[str, float]:
        """
        Classify soil type based on texture analysis using USDA soil texture triangle.
        
        Args:
            soil_properties: Dictionary containing clay, sand, silt percentages
            
        Returns:
            Tuple[str, float]: (soil_type, confidence)
        """
        if not soil_properties:
            return "Loamy", 0.25
        
        clay = soil_properties.get("clay", 0.0)
        sand = soil_properties.get("sand", 0.0)
        silt = soil_properties.get("silt", 0.0)
        
        # Enhanced texture classification based on USDA soil texture triangle
        
        # High clay content
        if clay >= 40:
            return "Clayey", 0.80
        
        # High sand content
        if sand >= 70:
            return "Sandy", 0.75
        
        # High silt content
        if silt >= 80:
            return "Silty", 0.75
        
        # Moderate clay (20-40%)
        if 20 <= clay < 40:
            if sand >= 45:
                return "Sandy", 0.65  # Sandy clay loam tendency
            else:
                return "Clayey", 0.65  # Clay loam tendency
        
        # Moderate silt (40-80%)
        if 40 <= silt < 80:
            if clay < 15:
                return "Silty", 0.70  # Silt loam
            else:
                return "Loamy", 0.65  # Silty clay loam
        
        # Default to loamy for balanced compositions
        return "Loamy", 0.60

    def get_location_info(self, latitude: float, longitude: float) -> Dict:
        """
        Get detailed location information using reverse geocoding.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            Dict: Location information with city, locality, region, country, formatted_address
        """
        try:
            url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "localityLanguage": "en"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract formatted address safely
            formatted_address = ""
            try:
                informative_data = data.get("localityInfo", {}).get("informative", [])
                if informative_data and isinstance(informative_data, list):
                    # Extract names from complex objects or use simple strings
                    address_parts = []
                    for item in informative_data:
                        if isinstance(item, dict):
                            name = item.get('name', '')
                            if name and len(address_parts) < 5:  # Limit to first 5 meaningful parts
                                address_parts.append(str(name))
                        elif isinstance(item, str) and item.strip():
                            address_parts.append(item.strip())
                    
                    formatted_address = ", ".join(address_parts[:5])  # Limit to 5 parts for readability
                
                # Fallback to basic address components if formatted address is empty
                if not formatted_address:
                    parts = []
                    if data.get("city"):
                        parts.append(data["city"])
                    if data.get("principalSubdivision"):
                        parts.append(data["principalSubdivision"])
                    if data.get("countryName"):
                        parts.append(data["countryName"])
                    formatted_address = ", ".join(parts)
                    
            except Exception as e:
                self.logger.warning(f"Error processing formatted address: {e}")
                # Create basic fallback address
                parts = []
                if data.get("city"):
                    parts.append(data["city"])
                if data.get("countryName"):
                    parts.append(data["countryName"])
                formatted_address = ", ".join(parts)
            
            location_info = {
                "city": data.get("city", ""),
                "locality": data.get("locality", ""),
                "region": data.get("principalSubdivision", ""),
                "country": data.get("countryName", ""),
                "formatted_address": formatted_address
            }
            
            self.logger.info(f"Location info retrieved for {latitude}, {longitude}: {location_info.get('formatted_address', 'Unknown')}")
            return location_info
            
        except Exception as e:
            self.logger.error(f"Reverse geocoding failed for {latitude}, {longitude}: {e}")
            return {
                "city": "",
                "locality": "",
                "region": "",
                "country": "",
                "formatted_address": ""
            }

# Create global instance
soil_data_api = SoilDataAPI()
