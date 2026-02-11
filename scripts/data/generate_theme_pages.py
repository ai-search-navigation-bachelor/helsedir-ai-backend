"""
Generate theme pages JSON from paths.

This script takes a list of theme page paths and generates a JSON file
with structured theme page data including titles, parent relationships,
and links.
"""

import json
import re
import uuid
from typing import Dict, List, Optional
from urllib.parse import unquote


def path_to_title(path: str) -> str:
    """
    Convert a URL path to a human-readable title.

    Examples:
        /digitalisering-og-e-helse -> Digitalisering og e-helse
        /autorisasjon-og-spesialistutdanning -> Autorisasjon og spesialistutdanning
    """
    # Get last segment
    segments = [s for s in path.split('/') if s]
    if not segments:
        return ""

    last_segment = segments[-1]

    # URL decode (handles %C3%B8 -> ø, etc.)
    last_segment = unquote(last_segment)

    # Replace hyphens with spaces
    title = last_segment.replace('-', ' ')

    # Capitalize first letter of each word, but preserve acronyms
    words = title.split()
    capitalized_words = []
    for word in words:
        # Keep acronyms like 'e-helse' lowercase
        if word in ['og', 'for', 'i', 'av', 'til', 'med', 'pa', 'om']:
            capitalized_words.append(word)
        else:
            # Capitalize first letter
            capitalized_words.append(word[0].upper() + word[1:] if len(word) > 1 else word.upper())

    return ' '.join(capitalized_words)


def get_parent_path(path: str) -> Optional[str]:
    """Get the parent path from a path."""
    segments = [s for s in path.split('/') if s]
    if len(segments) <= 1:
        return None
    return '/' + '/'.join(segments[:-1])


def get_children_paths(path: str, all_paths: List[str]) -> List[str]:
    """Get all direct children of a path."""
    children = []
    path_depth = path.count('/')

    for p in all_paths:
        if p == path:
            continue
        if p.startswith(path + '/'):
            # Check if it's a direct child (one level deeper)
            if p.count('/') == path_depth + 1:
                children.append(p)

    return sorted(children)


# Root theme categories with fixed category codes
ROOT_CATEGORIES = {
    "forebygging-diagnose-og-behandling": "0001",
    "digitalisering-og-e-helse": "0002",
    "lov-og-forskrift": "0003",
    "helseberedskap": "0004",
    "autorisasjon-og-spesialistutdanning": "0005",
    "tilskudd-og-finansiering": "0006",
    "statistikk-registre-og-rapporter": "0007",
}


def get_category_code(path: str) -> str:
    """Get category code for a path based on root theme."""
    segments = [s for s in path.split('/') if s]
    if not segments:
        return "9999"  # Uncategorized

    root = segments[0]
    return ROOT_CATEGORIES.get(root, "9999")  # 9999 for uncategorized


def generate_theme_pages(paths: List[str]) -> List[Dict]:
    """Generate theme page data from paths."""
    # First, decode all paths and replace spaces with hyphens
    decoded_paths = [unquote(p).replace(' ', '-') for p in paths]

    theme_pages = []

    for path in sorted(decoded_paths):
        # Determine which root category this page belongs to
        category_code = get_category_code(path)

        # Generate ID in same format as Helsedir API content
        # Format: 9999-{category}-{uuid}
        # - 9999: indicates theme page
        # - category: 0001-0007 for main themes, 9999 for uncategorized
        # - uuid: unique identifier for this specific page
        page_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://helsedirektoratet.no{path}"))
        page_id = f"9999-{category_code}-{page_uuid}"

        # Generate title
        title = path_to_title(path)

        # Get parent
        parent_path = get_parent_path(path)

        # Get children
        children = get_children_paths(path, decoded_paths)

        # Build links array
        links = []

        # Add parent link if exists
        if parent_path:
            links.append({
                "rel": "forelder",
                "type": "temaside",
                "href": parent_path
            })

        # Add child links
        for child_path in children:
            links.append({
                "rel": "barn",
                "type": "temaside",
                "href": child_path
            })

        theme_page = {
            "id": page_id,
            "tittel": title,
            "tekst": "",  # Empty for now
            "info_type": "temaside",
            "path": path,
            "koder": None,
            "maalgruppe": None,
            "links": links
        }

        theme_pages.append(theme_page)

    return theme_pages


# List of all theme page paths
THEME_PATHS = [
    "/autorisasjon-og-spesialistutdanning",
    "/autorisasjon-og-spesialistutdanning/autorisasjon-og-lisens",
    "/autorisasjon-og-spesialistutdanning/autorisasjon-og-lisens/sok-om-autorisasjon-og-lisens/avgj%C3%B8relse%20om%20gjennomf%C3%B8ring%20av%20tilleggskrav",
    "/autorisasjon-og-spesialistutdanning/certificate-of-current-professional-status-ccps",
    "/autorisasjon-og-spesialistutdanning/spesialistutdanning-og-godkjenning-for-leger",
    "/autorisasjon-og-spesialistutdanning/spesialistutdanning-og-godkjenning-for-leger/felles-kompetansemal-for-alle-deler-av-spesialistutdanningen",
    "/autorisasjon-og-spesialistutdanning/spesialistutdanning-og-godkjenning-for-leger/legespesialiteter",
    "/digitalisering-og-e-helse",
    "/digitalisering-og-e-helse/digital-samhandling",
    "/digitalisering-og-e-helse/e-helsestandarder-og-standardiseringstiltak",
    "/digitalisering-og-e-helse/e-helsestandarder-og-standardiseringstiltak/plan-for-internasjonale-e-helsestandarder",
    "/digitalisering-og-e-helse/e-helsestandarder-og-standardiseringstiltak/plan-for-internasjonale-e-helsestandarder/samarbeidsmodell-for-internasjonale-standarder/her-kan-du-engasjere-deg",
    "/digitalisering-og-e-helse/helsefaglig-terminologi",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/icd",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/icpc",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/kodeverk-i-e-helsestandarder",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/nkpk",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/nlk",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/norpat",
    "/digitalisering-og-e-helse/helsefaglige-kodeverk/phbu",
    "/digitalisering-og-e-helse/kritisk-informasjon-i-kjernejournal",
    "/digitalisering-og-e-helse/kunstig-intelligens",
    "/digitalisering-og-e-helse/kunstig-intelligens/rapporter-om-store-sprakmodeller-i-helse-og-omsorgstjenesten",
    "/digitalisering-og-e-helse/nasjonal-arkitektur",
    "/digitalisering-og-e-helse/nasjonal-e-helseportefolje",
    "/digitalisering-og-e-helse/nasjonal-e-helseportefolje/tiltak-i-nasjonal-e-helseportefolje",
    "/digitalisering-og-e-helse/nasjonal-e-helsestrategi",
    "/digitalisering-og-e-helse/normen-personvern-og-informasjonssikkerhet",
    "/digitalisering-og-e-helse/normen-personvern-og-informasjonssikkerhet/normen",
    "/digitalisering-og-e-helse/pasientens-journaldokumenter-pjd",
    "/digitalisering-og-e-helse/pasientens-legemiddelliste",
    "/digitalisering-og-e-helse/plan-for-digitalisering-pa-legemiddelomradet",
    "/digitalisering-og-e-helse/prinsipper-for-innbyggertjenester",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/anbefalinger-om-internasjonale-e-helsestandarder",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/arkitektur-og-informasjonssikkerhet",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/e-reseptstandarder",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/epj-helsedata-og-avlevering-til-norsk-helsearkiv",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/grunnleggende-standarder-for-elektronisk-samhandling",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/obligatoriske-og-anbefalte-kodeverk",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/samhandling-med-nav-helfo-og-helsenorge",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/standarder-for-elektronisk-samhandling-i-pasientforlop",
    "/digitalisering-og-e-helse/referansekatalogen-for-e-helse/standarder-for-rapportering-til-nasjonale-registre",
    "/digitalisering-og-e-helse/reguleringsplanen",
    "/digitalisering-og-e-helse/utviklingstrekk-2025",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/apotek",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/avtalespesialister",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/fastlege-og-apotek",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/folkehelseinstituttet",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helse-midt-norge-rhf",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helse-nord-rhf",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helse-og-omsorgsdepartementet",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helse-sor-ost-rhf",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helse-vest-rhf",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/helsedirektoratet",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/kommuner-og-fastleger",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/kommuner-og-fylkeskommuner",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/ks-og-kommuner",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/norsk-helsenett",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/regionale-helseforetak",
    "/digitalisering-og-e-helse/veikart-for-nasjonal-e-helsestrategi/spesialisthelsetjenesten-inkl.avtalespesialister",
    "/english",
    "/forebygging-diagnose-og-behandling",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/antibiotika",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/cfs-me",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/demens",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/diabetes",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/hjerneslag",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/hjerte-og-karsykdommer",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kols",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/koronavirus",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/analkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/blaerekreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/brystkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/bukspyttkjertelkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/cancer-thyroidea-skjoldbruskkjertelkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/diagnostisk-pakkeforlop",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/foflekkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/galleveiskreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/gynekologisk-kreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/hjernesvulst-og-hjernekreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/hode-hals-kreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/kreft-hos-barn",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/lungekreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/maligne-blodsykdommer",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/metastaser",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/nevroendokrine-svulster",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/nyrekreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/peniskreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/primaer-leverkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/prostatakreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/sarkom",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/spiserors-og-magesekkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/testikkelkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/tykk-og-endetarmskreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/kreft/tynntarmkreft",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/legemidler",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/legemidler/legemiddelbruk",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/legemidler/legemiddelfinansiering",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/legemidler/rekvirering-av-legemidler",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/multippel-sklerose",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/overvekt-og-fedme",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/palliasjon",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/persontilpasset-medisin",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/syn-og-horsel",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/underernaering",
    "/forebygging-diagnose-og-behandling/diagnose-og-behandling/utviklingshemming",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/alkohol",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/arbeid-og-helse",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/folkehelsearbeid-i-kommunen",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/folkehelsestatistikk-og-profiler",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/folkehelsestatistikk-og-profiler/last-ned-folkehelseprofil-eller-oppvekstprofil",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/fysisk-aktivitet",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/helsekompetanse",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/kosthold-og-ernaering",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/kosthold-og-ernaering/mat-og-maltider-i-skole-og-barnehage",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/livskvalitet-trivsel-og-folkehelsearbeid",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/miljo-og-helse",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/miljo-og-helse/klimaendringer",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/miljo-og-helse/skadedyr",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/motiverende-intervju-mi",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/seksuell-helse",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/smittevern",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/sosial-ulikhet-i-helse",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/sovn",
    "/forebygging-diagnose-og-behandling/forebygging-og-levevaner/tobakk",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/akuttmedisin",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/barnevern",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/blodgivning-og-transfusjonsmedisin",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/fastlegeordningen-legevakt-og-andre-allmennlegetjenester",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/fengselshelsetjenester",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/frisklivssentraler",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/helsefellesskap",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/helsestasjons-og-skolehelsetjenesten",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/innvandrere-asylsokere-og-flyktninger",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/kompetanseloft-2025",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/pasientsikkerhet-og-kvalitetsforbedring",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/personell-og-kompetanse",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/prioritering-i-helsetjenesten",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/rehabilitering-habilitering-og-individuell-plan",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/sykehjem-og-hjemmetjenester",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/sykehjem-og-hjemmetjenester/forebygging-kultur-og-aktiv-omsorg",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/tannhelse",
    "/forebygging-diagnose-og-behandling/organisering-og-tjenestetilbud/velferdsteknologi-og-digital-hjemmeoppfolging",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/adhd",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/angst-og-depresjon",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/kontrollkommisjonen-for-psykisk-helsevern",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/lokalt-psykisk-helse-og-rusarbeid",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/psykose",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/rus-og-avhengighet",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/selvskading-og-selvmord",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/spiseforstyrrelser",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/tvang-i-psykisk-helsevern",
    "/forebygging-diagnose-og-behandling/psykisk-helse-rus-og-avhengighet/vold-og-overgrep",
    "/forebygging-diagnose-og-behandling/svangerskap-fodsel-og-barsel",
    "/forebygging-diagnose-og-behandling/svangerskap-fodsel-og-barsel/abort",
    "/forebygging-diagnose-og-behandling/svangerskap-fodsel-og-barsel/amming-og-spedbarnsernaering",
    "/forebygging-diagnose-og-behandling/svangerskap-fodsel-og-barsel/assistert-befruktning",
    "/forebygging-diagnose-og-behandling/svangerskap-fodsel-og-barsel/svangerskap-fodsel-og-barsel",
    "/helseberedskap",
    "/helseberedskap/psykososial-oppfolging-ved-ulykker-kriser-og-katastrofer",
    "/helseberedskap/ukraina",
    "/lov-og-forskrift",
    "/lov-og-forskrift/alternativ-behandling",
    "/lov-og-forskrift/bioteknologi",
    "/lov-og-forskrift/bioteknologi/bioreferansegruppa",
    "/lov-og-forskrift/bioteknologi/fosterdiagnostikk",
    "/lov-og-forskrift/digital-deling-av-helseopplysninger",
    "/lov-og-forskrift/folketrygdloven-kapittel-5",
    "/lov-og-forskrift/forerkort",
    "/lov-og-forskrift/genteknologi",
    "/lov-og-forskrift/helse-og-omsorgstjenesteloven",
    "/lov-og-forskrift/helseforskningsloven",
    "/lov-og-forskrift/helsepersonelloven",
    "/lov-og-forskrift/organer",
    "/lov-og-forskrift/parorende",
    "/lov-og-forskrift/pasient-og-brukerrettighetsloven",
    "/lov-og-forskrift/psykisk-helsevernloven",
    "/lov-og-forskrift/rett-til-valg-av-behandlingssted",
    "/lov-og-forskrift/spesialisthelsetjenesteloven",
    "/lov-og-forskrift/sterilisering",
    "/lov-og-forskrift/tannhelsetjenesteloven",
    "/lov-og-forskrift/taushetsplikt-og-opplysningsplikt",
    "/nasjonale-krav-og-anbefalinger",
    "/om-oss",
    "/statistikk-registre-og-rapporter",
    "/statistikk-registre-og-rapporter/helsedata-og-helseregistre",
    "/tilskudd",
    "/tilskudd-og-finansiering",
    "/tilskudd-og-finansiering/finansiering",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/drg-systemet",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/innsatsstyrt-finansiering-isf",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/kostnadsvekter",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/norsk-pasientklassifisering-npk",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/saerkoder",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/stg-systemet",
    "/tilskudd-og-finansiering/finansiering/innsatsstyrt-finansiering-og-drg-systemet/tfg-systemet",
    "/tilskudd/abc-for-god-psykisk-helse-fylkeskommunal-innsats-for-a-oke-befolkningens-kompetanse-om-a-ivareta-og-styrke-den-psykiske-helsen",
    "/tilskudd/aktivitetstilbud-for-personer-med-rusmiddelproblemer-psykiske-helseproblemer-og-eller-erfaring-fra-salg-og-bytte-av-seksuelle-tjenester",
    "/tilskudd/aktivitetstiltak-for-a-motvirke-ensomhet-og-passivitet",
    "/tilskudd/Behandling-og-rehabilitering-rettet-mot-arbeid-utdanning-og-meningsfull-aktivitet",
    "/tilskudd/bruker-og-parorendearbeid-innen-rus-og-psykisk-helsefeltet",
    "/tilskudd/brukerstyrte-hus-pa-rusmiddelfeltet",
    "/tilskudd/demensplan-2025--fagutvikling-og-kompetansehevende-tiltak",
    "/tilskudd/diabetesarbeid",
    "/tilskudd/digital-hjemmeoppfolging-av-kronisk-syke",
    "/tilskudd/drift-av-paraplyorganisasjoner-innen-psykisk-helse-rusmiddel-og-voldsfeltet",
    "/tilskudd/driftstilskudd-nasjonal-grunnstotte-til-frivillige-rusmiddelpolitiske-organisasjoner",
    "/tilskudd/etablering-av-stillinger-for-spesialister-i-pedodonti",
    "/tilskudd/etablering-og-utvikling-av-kommunale-frisklivs-laerings-og-mestringstilbud",
    "/tilskudd/fagutvikling-og-kompetanseheving-i-omsorgstjenestene-til-samiske-brukere",
    "/tilskudd/faktainformasjon-om-rus-og-rusmidler",
    "/tilskudd/finn-tilskudd",
    "/tilskudd/fontenehus-og-fontenehus-norge",
    "/tilskudd/forebyggende-innsats-pa-rusmiddel-og-dopingomradet",
    "/tilskudd/fremme-av-erfaringskompetanse-i-tjenestene-innen-psykisk-helse-rusmiddel-og-voldsfeltet",
    "/tilskudd/frivillig-informasjons-og-kontaktskapende-arbeid",
    "/tilskudd/frivillig-rusmiddelforebyggende-og-spilleavhengighetsforebyggende-innsats",
    "/tilskudd/grunntilskudd-til-institusjonsbaserte-tilbud-for-personer-med-rusmiddelproblemer-og-eller-erfaring-fra-salg-og-bytte-av-seksuelle-tjenester",
    "/tilskudd/helhetlig-stotte-til-parorende-med-krevende-omsorgsoppgaver",
    "/tilskudd/historiske-pensjonskostnader",
    "/tilskudd/hjelpetelefon-og-chattilbud-innen-psykisk-helse-rus-og-voldsfeltet",
    "/tilskudd/hjelpetiltak-for-personer-med-problematisk-spilleadferd",
    "/tilskudd/hva-er-tilskudd",
    "/tilskudd/ikt-opplaering-for-a-heve-den-digitale-kompetansen-blant-eldre",
    "/tilskudd/institutter-og-foreninger-som-driver-etter-videre-og-spesialistutdanning-i-psykisk-helse",
    "/tilskudd/integrert-lop-i-dobbeltkompetanse-for-tannleger",
    "/tilskudd/kommunalt-kompetanse-og-innovasjonstilskudd",
    "/tilskudd/kommunalt-rusarbeid",
    "/tilskudd/kommunalt%20-rusarbeid",
    "/tilskudd/kompetansehevende-tiltak-for-lindrende-behandling-og-omsorg-ved-livets-slutt",
    "/tilskudd/kompetansehevende-tiltak-i-tjenestene-til-personer-med-utviklingshemning",
    "/tilskudd/kunnskap-og-informasjon-om-lindrende-behandling-og-omsorg-ved-livets-slutt-for-barn-og-ungdom",
    "/tilskudd/kunnskapsbaserte-mestringskurs--opplaering-av-ansatte-i-helsetjenesten-og-nav",
    "/tilskudd/lag-en-god-soknad",
    "/tilskudd/lindrende-enheter",
    "/tilskudd/lis1-stillinger-tidligere-kommuneturnus-tilskudd-til-kommuner",
    "/tilskudd/lonn-for-fylkeskommunalt-ansatte-tannpleiere-under-videreutdanning",
    "/tilskudd/lonn-til-kandidater-under-odontologisk-spesialistutdanning",
    "/tilskudd/lonnstilskudd-for-masterutdanning-i-avansert-klinisk-allmennsykepleie",
    "/tilskudd/mobilisering-mot-ensomhet",
    "/tilskudd/modellutvikling-klinisk-ernaeringsfysiolog-som-ressurs-for-omsorgstjenesten",
    "/tilskudd/narkotikaprogram-med-domstolskontroll",
    "/tilskudd/nasjonal-alis-og-veiledning",
    "/tilskudd/nasjonale-tiltak-for-forebygging-av-selvmord-og-selvskading",
    "/tilskudd/nettverk-for-innforing-av-helseteknologi-og-prioriterte-samhandlingstiltak-i-kommunal-sektor",
    "/tilskudd/oppfolging-av-barn-og-unge-med-psykiske-helseutfordringer-og-rusmiddelrelaterte-problemer",
    "/tilskudd/oppfolging-av-voksne-med-behov-for-langvarig-oppfolging-og-sammensatte-tjenester",
    "/tilskudd/oppfolgingstilbud-til-personer-med-rusmiddelproblemer-psykiske-helseproblemer-og-eller-erfaring-fra-salg-og-bytte-av-seksuelle-handlinger",
    "/tilskudd/parorendeskoler-og-samtalegrupper",
    "/tilskudd/pilotprosjekt-for-samhandling-mellom-nav-og-kommunale-helsetjenester",
    "/tilskudd/primaerhelseteam",
    "/tilskudd/psykisk-helse-i-skolen",
    "/tilskudd/psykisk-helse-og-livskvalitet-kjonns-og-seksualitetsmangfold-lhbt-",
    "/tilskudd/radgivnings-stotte-og-veiledningstjenester-innen-psykisk-helse-rus-og-vold",
    "/tilskudd/rapportere-pa-tilskudd",
    "/tilskudd/saerlig-ressurskrevende-helse-og-omsorgstjenester",
    "/tilskudd/samtale-og-mestringstilbud-innen-psykisk-helse-rus-og-voldsfeltet",
    "/tilskudd/sekretariatsfunksjon-for-kommunene-i-helsefellesskap",
    "/tilskudd/seksuell-helse",
    "/tilskudd/spesialistutdanning-av-tannleger",
    "/tilskudd/standardvilkar-for-tilskot-fra-helsedirektoratet",
    "/tilskudd/studenter--psykisk-helse-og-rusmiddelbruk",
    "/tilskudd/styrket-arbeid-med-forebygging-tidlig-intervensjon-og-behandling-av-spiseforstyrrelser",
    "/tilskudd/styrking-av-legevakttjenesten-i-distriktskommuner",
    "/tilskudd/styrking-og-utvikling-av-helsestasjons-og-skolehelsetjenesten",
    "/tilskudd/tannhelsetilbud-til-tortur-og-overgrepsutsatte-og-personer-med-odontofobi",
    "/tilskudd/teknologisk-stotte-i-fritidsaktiviteter-for-barn-og-unge-med-nedsatt-funksjonsevne-og-deres-familier",
    "/tilskudd/testing-og-behandling-i-karantenehotell",
    "/tilskudd/tilrettelegging-for-a-ta-i-bruk-teknologi-i-den-kommunale-helse-og-omsorgstjenesten",
    "/tilskudd/tilskudd-til-arbeid-innen-feltet-hiv-og-seksuelt-overforbare-infeksjoner",
    "/tilskudd/tilskuddsprosessen",
    "/tilskudd/turnus-for-leger-og-fysioterapeuter-reise-og-flytteutgifter",
    "/tilskudd/turnustjeneste-for-kiropraktorer-tilskudd-for-veiledning",
    "/tilskudd/Utproving-og-innforing-av-digitale-samhandlingslosninger",
    "/tilskudd/veiledning-av-studenter-og-ansatte",
    "/tilskudd/veiledning-til-innlogging-i-helsedirektoratets-tilskuddsportal",
    "/tilskudd/vertskommunene",
    "/tilskudd/videreutvikling-av-digi-ung-chat",
]


if __name__ == "__main__":
    # Generate theme pages
    theme_pages = generate_theme_pages(THEME_PATHS)

    # Write to JSON file
    output_path = "data/theme_pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(theme_pages, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(theme_pages)} theme pages")
    print(f"Saved to {output_path}")

    # Print root category mapping
    print(f"\nRoot categories ({len(ROOT_CATEGORIES)}):")
    for root, code in sorted(ROOT_CATEGORIES.items(), key=lambda x: x[1]):
        print(f"  9999-{code}: /{root}")
    print(f"  9999-9999: /... (uncategorized)")

    # Print some examples
    print("\nExample theme pages:")
    for page in theme_pages[:3]:
        print(f"\nID: {page['id']}")
        print(f"Path: {page['path']}")
        print(f"Title: {page['tittel']}")
        print(f"Links: {len(page['links'])}")
