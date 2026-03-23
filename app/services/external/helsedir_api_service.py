"""
Service for interacting with Helsedirektoratet API.

API Documentation: https://utvikler.helsedirektoratet.no
"""
import httpx
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from app.config import settings
from app.exceptions.helsedir import HelseDirectorateAPIError

# Re-export for backwards compatibility
__all__ = ["HelseDirectorateAPIService", "HelseDirectorateAPIError", "helsedir_api_service"]


class HelseDirectorateAPIService:
    """
    Service for making API calls to Helsedirektoratet.

    All requests automatically include the required 'Ocp-Apim-Subscription-Key' header.
    """

    def __init__(self):
        self.base_url = settings.helsedir_api_url
        self.api_key = settings.helsedir_api_key
        self.nki_api_key = settings.helsedir_nki_api_key or settings.helsedir_api_key

        # Validate that API key is configured
        if not self.api_key:
            print("WARNING: HELSEDIR_API_KEY not configured in .env")
        if not self.nki_api_key:
            print("WARNING: HELSEDIR_NKI_API_KEY not configured in .env")

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """Get required headers for API requests."""
        return {
            "Ocp-Apim-Subscription-Key": api_key or self.api_key,
            "Accept": "application/json",
        }

    def _parse_nki_response(
        self,
        response: httpx.Response,
        *,
        indicator_id: Optional[str] = None,
        allow_no_content: bool = False,
    ) -> Optional[Any]:
        """Validate common NKI responses and return parsed JSON."""
        if allow_no_content and response.status_code == 204:
            return None
        if response.status_code == 401:
            raise HelseDirectorateAPIError("Unauthorized: Invalid API key")
        if response.status_code == 403:
            raise HelseDirectorateAPIError(
                "Forbidden: API key does not have access to this resource"
            )
        if response.status_code == 404:
            if allow_no_content and indicator_id:
                raise HelseDirectorateAPIError(f"NKI indicator data not found: {indicator_id}")
            if indicator_id:
                raise HelseDirectorateAPIError(f"NKI indicator not found: {indicator_id}")
            raise HelseDirectorateAPIError("NKI resource not found")
        if response.status_code >= 400:
            raise HelseDirectorateAPIError(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        response.raise_for_status()
        return response.json()

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

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

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

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    def get_file_by_id(
        self,
        file_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get a specific file payload by ID.

        Args:
            file_id: The file content ID
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the file payload

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/filer/{file_id}"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

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
                        f"File not found: {file_id}"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    def list_nki_quality_indicators(
        self,
        timeout: float = 15.0,
    ) -> Any:
        """
        List NKI quality indicators.

        Returns:
            Parsed JSON payload from /innhold/nki/kvalitetsindikatorer
        """
        if not self.nki_api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_NKI_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/nki/kvalitetsindikatorer"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(self.nki_api_key),
                    timeout=timeout,
                )
                return self._parse_nki_response(response)
        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    def get_nki_quality_indicator_by_id(
        self,
        indicator_id: str,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Get a single NKI quality indicator metadata payload.
        """
        if not self.nki_api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_NKI_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/nki/kvalitetsindikatorer/{indicator_id}"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(self.nki_api_key),
                    timeout=timeout,
                )
                return self._parse_nki_response(response, indicator_id=indicator_id)
        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    def get_nki_quality_indicator_data(
        self,
        indicator_id: str,
        timeout: float = 20.0,
        content_type: str = "application/json",
    ) -> Optional[Any]:
        """
        Get a single NKI quality indicator dataset payload.

        Returns:
            Parsed JSON payload, or None when the endpoint responds with 204.
        """
        if not self.nki_api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_NKI_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/nki/kvalitetsindikatorer/{indicator_id}/data"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers=self._get_headers(self.nki_api_key),
                    params={"contentType": content_type},
                    timeout=timeout,
                )
                return self._parse_nki_response(
                    response,
                    indicator_id=indicator_id,
                    allow_no_content=True,
                )
        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    async def get_infobit_by_id_async(
        self,
        infobit_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get a specific infobit by ID (async version for FastAPI).

        Args:
            infobit_id: The infobit ID (format: 0006-0014-70b46b52-eb30-4ee9-b8c8-ef5e238c419f)
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the full infobit data with all details

        Raises:
            HelseDirectorateAPIError: If the API request fails

        Example:
            >>> result = await service.get_infobit_by_id_async(
            ...     "0006-0014-70b46b52-eb30-4ee9-b8c8-ef5e238c419f"
            ... )
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/infobit/{infobit_id}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

                # Check for errors
                if response.status_code == 401:
                    raise HelseDirectorateAPIError(
                        "Unauthorized: Invalid API key"
                    )
                elif response.status_code == 404:
                    raise HelseDirectorateAPIError(
                        f"Infobit not found: {infobit_id}"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    async def get_kapittel_by_id_async(
        self,
        kapittel_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get a specific kapittel (chapter) by ID (async version for FastAPI).

        Args:
            kapittel_id: The kapittel ID
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the kapittel data

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/kapitler/{kapittel_id}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

                if response.status_code == 404:
                    raise HelseDirectorateAPIError(
                        f"Kapittel not found: {kapittel_id}"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    async def get_file_by_id_async(
        self,
        file_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get a specific file payload by ID (async version for FastAPI).

        Args:
            file_id: The file content ID
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the file payload

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/filer/{file_id}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

                if response.status_code == 401:
                    raise HelseDirectorateAPIError(
                        "Unauthorized: Invalid API key"
                    )
                elif response.status_code == 403:
                    raise HelseDirectorateAPIError(
                        "Forbidden: API key does not have access to this resource"
                    )
                elif response.status_code == 404:
                    raise HelseDirectorateAPIError(
                        f"File not found: {file_id}"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    async def get_content_by_href_async(
        self,
        href_url: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Get content from Helsedirektoratet API by full href URL.
        
        This is a generic method that can fetch any content type:
        - kapitler
        - pakkeforlop-anbefalinger
        - infobit
        - etc.

        Args:
            href_url: Full API URL (e.g., https://api.helsedirektoratet.no/innhold/...)
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing the content data

        Raises:
            HelseDirectorateAPIError: If the API request fails
        """
        if not self.api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_API_KEY not configured. Add it to your .env file."
            )

        # Validate that URL is from the correct API (using proper URL parsing)
        expected_parsed = urlparse(self.base_url)
        actual_parsed = urlparse(href_url)
        
        if (actual_parsed.scheme != expected_parsed.scheme or 
            actual_parsed.netloc != expected_parsed.netloc):
            raise HelseDirectorateAPIError(
                f"Invalid href URL: must be from {expected_parsed.scheme}://{expected_parsed.netloc}"
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    href_url,
                    headers=self._get_headers(),
                    timeout=timeout,
                )

                if response.status_code == 404:
                    raise HelseDirectorateAPIError(
                        f"Content not found: {href_url}"
                    )
                elif response.status_code >= 400:
                    raise HelseDirectorateAPIError(
                        f"API request failed with status {response.status_code}"
                    )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            )
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            )

    async def get_nki_quality_indicator_by_id_async(
        self,
        indicator_id: str,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Async version of get_nki_quality_indicator_by_id.
        """
        if not self.nki_api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_NKI_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/nki/kvalitetsindikatorer/{indicator_id}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(self.nki_api_key),
                    timeout=timeout,
                )
                return self._parse_nki_response(response, indicator_id=indicator_id)
        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

    async def get_nki_quality_indicator_data_async(
        self,
        indicator_id: str,
        timeout: float = 20.0,
        content_type: str = "application/json",
    ) -> Optional[Any]:
        """
        Async version of get_nki_quality_indicator_data.
        """
        if not self.nki_api_key:
            raise HelseDirectorateAPIError(
                "HELSEDIR_NKI_API_KEY not configured. Add it to your .env file."
            )

        url = f"{self.base_url}/innhold/nki/kvalitetsindikatorer/{indicator_id}/data"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(self.nki_api_key),
                    params={"contentType": content_type},
                    timeout=timeout,
                )
                return self._parse_nki_response(
                    response,
                    indicator_id=indicator_id,
                    allow_no_content=True,
                )
        except httpx.TimeoutException as e:
            raise HelseDirectorateAPIError(
                f"API request timed out after {timeout} seconds"
            ) from e
        except httpx.RequestError as e:
            raise HelseDirectorateAPIError(
                f"API request failed: {e!s}"
            ) from e

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
                f"API request failed: {e!s}"
            )


# Global instance
helsedir_api_service = HelseDirectorateAPIService()
