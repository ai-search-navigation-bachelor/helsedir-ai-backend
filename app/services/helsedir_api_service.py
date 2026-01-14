"""
Service for interacting with Helsedirektoratet API.

API Documentation: https://utvikler.helsedirektoratet.no
"""
import httpx
from typing import Optional, Dict, Any, List
from app.config import settings


class HelseDirectorateAPIError(Exception):
    """Custom exception for Helsedirektoratet API errors."""
    pass


class HelseDirectorateAPIService:
    """
    Service for making API calls to Helsedirektoratet.

    All requests automatically include the required 'Ocp-Apim-Subscription-Key' header.
    """

    def __init__(self):
        self.base_url = settings.helsedir_api_url
        self.api_key = settings.helsedir_api_key

        # Validate that API key is configured
        if not self.api_key:
            print("WARNING: HELSEDIR_API_KEY not configured in .env")

    def _get_headers(self) -> Dict[str, str]:
        """Get required headers for API requests."""
        return {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
        }

    def search_infobits(
        self,
        query_text: Optional[str] = None,
        filter_query: Optional[str] = None,
        search_mode: Optional[str] = None,
        query_type: Optional[str] = None,
        get_full_infobits: bool = False,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Search for infobits in Helsedirektoratet content.

        Args:
            query_text: The search query text
            filter_query: OData filter expression (e.g., "contentType eq 'veileder'")
            search_mode: Search mode ("any" or "all")
            query_type: Query type ("simple" or "full")
            get_full_infobits: Whether to return full infobit content
            timeout: Request timeout in seconds

        Returns:
            List of search results from the API (note: returns list directly, not dict)

            API uses Norwegian field names:
            - tittel (title)
            - tekst (text/body)
            - maalgruppe (target groups)
            - infoType (content type)

        Raises:
            HelseDirectorateAPIError: If the API request fails

        Example:
            >>> service = helsedir_api_service
            >>> results = service.search_infobits(
            ...     query_text="diabetes",
            ...     filter_query="targetGroups/any(t: t eq 'fastlege')",
            ...     get_full_infobits=True
            ... )
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        # Build query parameters
        params = {}
        if query_text:
            params["QueryText"] = query_text
        if filter_query:
            params["Filter"] = filter_query
        if search_mode:
            params["SearchMode"] = search_mode
        if query_type:
            params["QueryType"] = query_type
        if get_full_infobits:
            params["getFullInfobits"] = "true"

        # Make API request
        url = f"{self.base_url}/innhold/sok/infobit"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=timeout,
                )

                # Check for errors
                if response.status_code == 401:
                    raise HelseDirectorateAPIError(
                        "Unauthorized: Invalid API key. Check HELSEDIR_API_KEY in .env"
                    )
                elif response.status_code == 403:
                    raise HelseDirectorateAPIError(
                        "Forbidden: API key does not have access to this resource"
                    )
                elif response.status_code == 404:
                    raise HelseDirectorateAPIError(
                        "Not found: API endpoint does not exist"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            )
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {str(e)}"
            )

    def get_infobit_by_id(
        self,
        infobit_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get a specific infobit by ID.

        Args:
            infobit_id: The infobit ID
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the infobit data

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/infobit/{infobit_id}"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            )
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {str(e)}"
            )

    async def search_infobits_async(
        self,
        query_text: Optional[str] = None,
        filter_query: Optional[str] = None,
        search_mode: Optional[str] = None,
        query_type: Optional[str] = None,
        get_full_infobits: bool = False,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Async version of search_infobits for use in FastAPI endpoints.

        Args:
            query_text: The search query text
            filter_query: OData filter expression
            search_mode: Search mode ("any" or "all")
            query_type: Query type ("simple" or "full")
            get_full_infobits: Whether to return full infobit content
            timeout: Request timeout in seconds

        Returns:
            List of search results from the API (note: returns list directly, not dict)

            API uses Norwegian field names:
            - tittel (title)
            - tekst (text/body)
            - maalgruppe (target groups)
            - infoType (content type)

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        # Build query parameters
        params = {}
        if query_text:
            params["QueryText"] = query_text
        if filter_query:
            params["Filter"] = filter_query
        if search_mode:
            params["SearchMode"] = search_mode
        if query_type:
            params["QueryType"] = query_type
        if get_full_infobits:
            params["getFullInfobits"] = "true"

        url = f"{self.base_url}/innhold/sok/infobit"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=timeout,
                )

                # Check for errors
                if response.status_code == 401:
                    raise HelseDirectorateAPIError(
                        "Unauthorized: Invalid API key"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            )
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {str(e)}"
            )


# Global instance
helsedir_api_service = HelseDirectorateAPIService()
