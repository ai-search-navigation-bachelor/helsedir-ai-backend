#!/usr/bin/env python3
"""
Test NTNU HPC LLM API connection and functionality.

Verifies that the API key is configured correctly and the model responds.

Usage:
    python scripts/test/test_ntnu_llm.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
import httpx


def test_api_connection():
    """Test basic API connection and configuration."""
    print("=" * 60)
    print("NTNU HPC LLM API TEST")
    print("=" * 60)

    # Check configuration
    print("\n1. Checking configuration...")
    print(f"   API URL: {settings.ntnu_llm_api_url}")
    print(f"   Model: {settings.ntnu_llm_model}")

    if not settings.ntnu_llm_api_key:
        print("   ❌ API Key: NOT CONFIGURED")
        print("\n   ERROR: NTNU_LLM_API_KEY not set in .env")
        print("   Please add your NTNU HPC API key to .env file")
        return False

    print(f"   ✅ API Key: {settings.ntnu_llm_api_key[:20]}...")

    # Test API call
    print("\n2. Testing API connection...")

    test_prompt = "Hva er diabetes type 2?"

    try:
        endpoint = f"{settings.ntnu_llm_api_url.rstrip('/')}/chat/completions"

        print(f"   Endpoint: {endpoint}")
        print(f"   Sending test prompt: '{test_prompt}'")

        response = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.ntnu_llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.ntnu_llm_model,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=30.0,
        )

        print(f"   Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"\n   📋 Full API response structure:")
            print(f"   Keys: {list(data.keys())}")

            # Extract response
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                print(f"   Choice keys: {list(choice.keys())}")

                message = choice.get("message", {})
                print(f"   Message keys: {list(message.keys())}")

                llm_response = message.get("content")

                print("\n   ✅ API CONNECTION SUCCESSFUL!")
                print("\n" + "=" * 60)
                print("LLM RESPONSE:")
                print("=" * 60)
                if llm_response:
                    print(llm_response)
                else:
                    print("(empty or None)")
                    print(f"\nFull message object: {message}")
                print("=" * 60)

                if not llm_response:
                    print("\n   ❌ Response content is empty or None")
                    return False

                # Check if response is in Norwegian
                norwegian_words = ["er", "og", "for", "med", "som", "diabetes"]
                has_norwegian = any(word in llm_response.lower() for word in norwegian_words)

                if has_norwegian:
                    print("\n   ✅ Response appears to be in Norwegian")
                else:
                    print("\n   ⚠️  Response may not be in Norwegian")

                return True
            else:
                print("\n   ❌ Unexpected response format")
                print(f"   Response: {data}")
                return False

        elif response.status_code == 401:
            print("\n   ❌ AUTHENTICATION FAILED")
            print("   Error: Invalid API key")
            print("   Please check your NTNU_LLM_API_KEY in .env")
            return False

        elif response.status_code == 404:
            print("\n   ❌ ENDPOINT NOT FOUND")
            print(f"   Error: {endpoint} does not exist")
            print("   Please check NTNU_LLM_API_URL in .env")
            return False

        elif response.status_code == 400:
            print("\n   ❌ BAD REQUEST")
            print(f"   Error: {response.text[:300]}")
            print("   The model name may be incorrect")
            print(f"   Current model: {settings.ntnu_llm_model}")
            return False

        else:
            print(f"\n   ❌ API ERROR: {response.status_code}")
            print(f"   Response: {response.text[:300]}")
            return False

    except httpx.TimeoutException:
        print("\n   ❌ REQUEST TIMED OUT")
        print("   The API did not respond within 30 seconds")
        print("   This may be a network issue or the API may be slow")
        return False

    except httpx.RequestError as e:
        print(f"\n   ❌ REQUEST ERROR: {e}")
        print("   Could not connect to the API")
        print("   Please check your network connection")
        return False

    except Exception as e:
        print(f"\n   ❌ UNEXPECTED ERROR: {e}")
        return False


def clean_query(q: str, num_queries: int) -> list:
    """
    Clean LLM output to extract queries (same logic as generate_queries.py).

    Args:
        q: Raw query text from LLM
        num_queries: Target number of queries

    Returns:
        List of cleaned query strings
    """
    import re

    # Remove <think> tags and content (NorwAI-Magistral reasoning model)
    # Handle both closed tags and unclosed tags
    text = re.sub(r'<think>.*?</think>', '', q, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)  # Unclosed tags

    # Remove standalone <think> or </think> tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)

    queries = [line.strip() for line in text.split("\n") if line.strip()]

    # Clean up and filter
    cleaned = []
    skip_phrases = [
        "her er",
        "søkeord",
        "fraser",
        "følgende",
        "liste",
        "kunne brukt",
        "ville brukt",
        "for å finne",
        "svar:",
        "tenke",
        "greit",
        "okei",
    ]

    for q in queries:
        # Remove markdown bold (**text**)
        q = re.sub(r'\*\*([^*]+)\*\*', r'\1', q)

        # Remove LaTeX/formatting - more aggressive
        q = re.sub(r'\\textbf\{([^}]+)\}', r'\1', q)  # \textbf{text}
        q = re.sub(r'\\text[a-z]*\{([^}]+)\}', r'\1', q)  # \text*, \textit, etc
        q = re.sub(r'\\[a-z]+\{([^}]+)\}', r'\1', q)  # Any \command{text}
        q = re.sub(r'[a-z]*tbf\{', '', q)  # Leftover like 'extbf{'
        q = re.sub(r'\\+', '', q)  # Remaining backslashes

        # Remove "Svar:" or similar prefixes
        q = re.sub(r'^(svar|søk|query|queries?):\s*', '', q, flags=re.IGNORECASE)

        # Remove numbering, bullets, and leading symbols
        q = re.sub(r'^[\d\.\-\)\*\•\s]+', '', q)

        # Remove "Søk etter" prefix (case insensitive)
        if q.lower().startswith("søk etter "):
            q = q[10:]

        # Strip quotes, braces, brackets, and whitespace
        q = q.strip('"\'{}[]()').strip()

        # Skip if empty after cleaning
        if not q:
            continue

        # Skip if only think tag remains
        if q.lower() in ['<think>', '</think>', 'think']:
            continue

        # Skip if too long (likely explanation text)
        if len(q) > 100:
            continue

        # Skip if contains colon (likely label)
        if ":" in q:
            continue

        # Skip if starts with incomplete quote
        if (q.startswith('"') and not q.endswith('"')) or \
           (q.startswith("'") and not q.endswith("'")):
            continue

        # Skip if contains common intro/thinking phrases
        q_lower = q.lower()
        if any(phrase in q_lower for phrase in skip_phrases):
            continue

        # Skip if it looks like reasoning/explanation
        if len(q.split()) > 10:  # More than 10 words is probably not a query
            continue

        cleaned.append(q)

        # Stop when we have enough
        if len(cleaned) >= num_queries:
            break

    return cleaned


def test_query_generation():
    """Test query generation and cleaning on multiple documents."""
    print("\n" + "=" * 60)
    print("3. Testing query generation and cleaning...")
    print("=" * 60)

    # Test cases with different document types (6 diverse examples)
    test_cases = [
        {
            "name": "Diabetes retningslinje",
            "passage": """Behandling av diabetes type 2. Dette er en retningslinje.
Målgruppe: fastlege, spesialist.
Relevante koder: E11, T90.
Diabetes type 2 er en kronisk sykdom hvor kroppen ikke klarer å regulere blodsukkeret
på en normal måte. Behandling inkluderer livsstilsendringer og eventuelt medikamenter.""",
            "info_type": "retningslinje"
        },
        {
            "name": "KOLS veileder",
            "passage": """Kronisk obstruktiv lungesykdom (KOLS). Dette er en veileder.
Målgruppe: fastlege, sykepleier.
Relevante koder: J44.
KOLS er en progressiv lungesykdom preget av vedvarende luftveisobstruksjon.
Røykeslutt er den viktigste behandlingen.""",
            "info_type": "veileder"
        },
        {
            "name": "Hjerteinfarkt anbefaling",
            "passage": """Akutt hjerteinfarkt - initial behandling. Dette er en anbefaling.
Målgruppe: akuttmedisin, kardiolog.
Relevante koder: I21.
Ved mistanke om hjerteinfarkt må pasienten umiddelbart få ASA og nitroglyserin.
Rask transport til PCI-senter er avgjørende.""",
            "info_type": "anbefaling"
        },
        {
            "name": "Depresjon pakkeforløp",
            "passage": """Pakkeforløp for depresjon. Dette er et pakkeforløp.
Målgruppe: fastlege, psykolog, psykiater.
Relevante koder: F32, F33, ICPC2: P76.
Depresjon er en vanlig psykisk lidelse. Pakkeforløpet sikrer rask utredning
og behandlingsstart. Behandling kan være medikamentell og/eller psykoterapi.""",
            "info_type": "pakkeforlop-anbefaling"
        },
        {
            "name": "Antibiotikabruk prosedyre",
            "passage": """Retningslinjer for antibiotikabruk. Dette er en prosedyre.
Målgruppe: fastlege, spesialist.
Antibiotikaresistens er en økende utfordring. Bruk smalspektret antibiotika
når mulig. Ved urinveisinfeksjon er nitrofurantoin førstevalg.""",
            "info_type": "prosedyre"
        },
        {
            "name": "Barn astma veileder",
            "passage": """Astma hos barn - diagnostikk og behandling. Dette er en veileder.
Målgruppe: fastlege, barnelege.
Relevante koder: J45, ICPC2: R96.
Astma er den vanligste kroniske sykdommen hos barn. Behandling tilpasses
alvorlighetsgrad. Inhalasjonssteroider er grunnbehandling ved vedvarende astma.""",
            "info_type": "veileder"
        }
    ]

    all_success = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'─' * 60}")

        prompt = f"""Du er en ekspert på norsk helseinformasjon.

SVAR PÅ NORSK!

Generer 5 realistiske søkeord/fraser som en helsepersonell ville brukt for å finne dette dokumentet.

Dokumenttype: {test_case['info_type']}
Dokumentinnhold:
{test_case['passage']}

VIKTIG:
- Ikke bruk <think> tags
- Ikke forklar resonneringen din
- Ikke skriv "Svar:" eller lignende
- BARE list opp søkefrasene

Regler for søkefrasene:
- Start hver søkefrase med stor forbokstav
- Skriv søk slik en lege eller sykepleier ville skrevet dem
- Bruk SPESIFIKKE detaljer fra dokumentet (koder, målgrupper, spesifikke tilstander)
- Bruk faktiske termer og koder fra teksten (f.eks. E11, J44, STEMI)
- Fokuser på KORTE, SPESIFIKKE søk (1-3 ord) - de fleste brukere søker kort
- Generer en mix: minst 3 veldig korte (1-2 ord), resten kan være 3-5 ord
- Ikke bruk generelle termer - vær spesifikk for DETTE dokumentet
- Ikke bruk anførselstegn eller nummerering

Returner BARE 5 søkefraser, én per linje, INGEN forklaring:"""

        try:
            endpoint = f"{settings.ntnu_llm_api_url.rstrip('/')}/chat/completions"

            # Show the full prompt being sent
            if i == 1:  # Only show for first test to avoid clutter
                print(f"\n   📤 FULL PROMPT BEING SENT TO API:")
                print("   " + "═" * 56)
                for line in prompt.split('\n'):
                    print(f"   {line}")
                print("   " + "═" * 56)

            response = httpx.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.ntnu_llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.ntnu_llm_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 200,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                raw_text = data["choices"][0]["message"]["content"].strip()

                # Show FULL raw output (no truncation)
                print(f"\n   📝 Raw LLM output (full):")
                print("   " + "─" * 56)
                for line in raw_text.split('\n'):
                    print(f"   {line}")
                print("   " + "─" * 56)

                # Clean queries using same logic as generate_queries.py
                cleaned_queries = clean_query(raw_text, num_queries=5)

                print(f"\n   ✅ Generated {len(cleaned_queries)} cleaned queries:")
                for j, q in enumerate(cleaned_queries, 1):
                    print(f"      {j}. {q}")

                if len(cleaned_queries) >= 3:
                    print(f"   ✅ Success!")
                else:
                    print(f"   ⚠️  Only got {len(cleaned_queries)} queries (expected 5)")
                    all_success = False

            else:
                print(f"\n   ❌ Failed: {response.status_code}")
                all_success = False

        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            all_success = False

    return all_success


def main():
    """Run all tests."""

    # Test 1: Basic connection
    success1 = test_api_connection()

    if not success1:
        print("\n" + "=" * 60)
        print("RESULT: ❌ BASIC CONNECTION FAILED")
        print("=" * 60)
        print("\nPlease fix the configuration issues above before proceeding.")
        sys.exit(1)

    # Test 2: Query generation
    success2 = test_query_generation()

    # Final result
    print("\n" + "=" * 60)
    if success1 and success2:
        print("RESULT: ✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nYour NTNU LLM API is configured correctly!")
        print("You can now run: python scripts/data/generate_queries.py")
    else:
        print("RESULT: ⚠️  PARTIAL SUCCESS")
        print("=" * 60)
        print("\nBasic connection works, but query generation had issues.")
        print("You may still be able to use the API.")

    print()


if __name__ == "__main__":
    main()
