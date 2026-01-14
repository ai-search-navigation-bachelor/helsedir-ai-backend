"""
Test script for Helsedirektoratet API integration.

Run this script to verify your API key and test the connection.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.helsedir_api_service import helsedir_api_service, HelseDirectorateAPIError
from app.config import settings


def test_api_connection():
    """Test connection to Helsedirektoratet API."""
    print("=" * 60)
    print("Testing Helsedirektoratet API Connection")
    print("=" * 60)
    print()

    # Check if API key is configured
    if not settings.helsedir_api_key:
        print("[ERROR] HELSEDIR_API_KEY not configured in .env")
        print()
        print("To fix:")
        print("1. Register at https://utvikler.helsedirektoratet.no")
        print("2. Subscribe to the API")
        print("3. Add your key to .env:")
        print("   HELSEDIR_API_KEY=your_key_here")
        print()
        return False

    print(f"[OK] API Key configured: {settings.helsedir_api_key[:8]}...")
    print(f"[OK] API URL: {settings.helsedir_api_url}")
    print()

    # Test 1: Simple search
    print("Test 1: Simple search for 'diabetes'")
    print("-" * 60)
    try:
        results = helsedir_api_service.search_infobits(
            query_text="diabetes",
            get_full_infobits=False,
        )

        print(f"[OK] Search successful!")
        print(f"     Results: {len(results)} items")

        # Show first result
        if results:
            first = results[0]
            print(f"     First result: {first.get('tittel', 'N/A')}")

    except HelseDirectorateAPIError as e:
        print(f"[ERROR] {e}")
        return False
    print()

    # Test 2: Search with filter
    print("Test 2: Search with role filter")
    print("-" * 60)
    try:
        results = helsedir_api_service.search_infobits(
            query_text="vaksine",
            filter_query="maalgruppe/any(t: t eq 'fastlege')",
            get_full_infobits=False,
        )

        print(f"[OK] Filtered search successful!")
        print(f"     Results for fastlege: {len(results)} items")

    except HelseDirectorateAPIError as e:
        print(f"[ERROR] {e}")
        return False
    print()

    # Test 3: Get full infobits
    print("Test 3: Get full infobit content")
    print("-" * 60)
    try:
        results = helsedir_api_service.search_infobits(
            query_text="smittevern",
            get_full_infobits=True,
        )

        print(f"[OK] Full content search successful!")
        print(f"     Results: {len(results)} items")

        # Check if we got full content
        if results:
            first = results[0]
            has_body = bool(first.get('tekst'))
            print(f"     Has full content: {has_body}")

            # Show field names
            print(f"     Available fields: {', '.join(list(first.keys()))}")

    except HelseDirectorateAPIError as e:
        print(f"[ERROR] {e}")
        return False
    print()

    print("=" * 60)
    print("[SUCCESS] All API tests passed!")
    print("=" * 60)
    print()
    print("You can now use the API in your application:")
    print()
    print("from app.services.helsedir_api_service import helsedir_api_service")
    print()
    print("results = helsedir_api_service.search_infobits(")
    print("    query_text='diabetes',")
    print("    filter_query=\"targetGroups/any(t: t eq 'fastlege')\",")
    print("    get_full_infobits=True")
    print(")")
    print()

    return True


if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
