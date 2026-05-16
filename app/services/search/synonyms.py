"""
Norwegian medical synonym dictionary for search expansion.

Covers common health topics from Helsedirektoratet content:
- Disease names (fagtermer ↔ folkelige termer)
- Abbreviations
- Norwegian ↔ Latin/English medical terms
- Common spelling variations

Each key maps to a list of synonyms. The mapping is bidirectional:
all terms in a synonym group can find each other.
"""

import re
from typing import Dict, FrozenSet, List, Set, Tuple

# ---------------------------------------------------------------------------
# Synonym groups – each tuple contains terms that are interchangeable.
# Order does not matter; all terms in a group map to each other.
# ---------------------------------------------------------------------------
_SYNONYM_GROUPS = [
    # ── Hjerte og kar ─────────────────────────────────────────────
    ("hjerteinfarkt", "myokardinfarkt", "akutt hjerteinfarkt", "myocardial infarction"),
    ("hjerneslag", "slag", "apopleksi", "cerebralt insult", "stroke"),
    ("hjertesvikt", "kardial svikt", "heart failure"),
    ("høyt blodtrykk", "hypertensjon"),
    ("lavt blodtrykk", "hypotensjon"),
    ("atrieflimmer", "forkammerflimmer", "atrial fibrillation", "af"),
    ("atrieflutter", "forkammerflutter", "atrial flutter"),
    ("arytmi", "hjerterytmeforstyrrelse", "rytmeforstyrrelse"),
    ("angina pectoris", "angina"),
    ("åreforkalkning", "aterosklerose", "arteriosklerose"),
    ("blodpropp", "trombose"),
    ("dyp venetrombose", "dvt", "deep vein thrombosis"),
    ("lungeemboli", "pulmonal emboli"),
    ("hjertestans", "cardiac arrest"),
    ("koronarsykdom", "kransåresykdom", "iskemisk hjertesykdom", "koronar hjertesykdom"),
    ("høyt kolesterol", "hyperkolesterolemi"),
    ("blodfortynnende", "antikoagulantia", "antikoagulasjon"),
    ("platehemmere", "trombocytthemmere", "platehemmende legemidler"),
    ("hjertesekkbetennelse", "perikarditt"),
    ("hjertemuskelbetennelse", "myokarditt"),
    ("hjerteklaffsykdom", "klaffefeil"),
    ("aortastenose", "forsnevring i aortaklaffen"),
    ("endokarditt", "hjerteklaffinfeksjon"),
    ("aneurisme", "utposning på blodåre", "karutposning"),
    ("aortaaneurisme", "utvidelse av hovedpulsåren"),
    ("åreknuter", "varicer", "varikøse vener"),
    ("betennelse i blodåre", "vaskulitt", "karbetennelse"),
    ("hjertekrampe", "angina pectoris"),
    ("hjertebank", "palpitasjoner", "palpitationer"),
    ("besvimelse", "synkope"),
    ("hevelse i beina", "beinødem", "perifere ødemer"),

    # ── Diabetes og metabolsk ─────────────────────────────────────
    ("diabetes", "diabetes mellitus", "sukkersyke"),
    ("diabetes type 1", "type 1 diabetes", "t1d", "dm1"),
    ("diabetes type 2", "type 2 diabetes", "t2d", "dm2"),
    ("svangerskapsdiabetes", "gestasjonell diabetes"),
    ("blodsukker", "blodglukose", "glukose"),
    ("langtidsblodsukker", "hba1c", "glykolisert hemoglobin"),
    ("insulinresistens", "nedsatt insulinfølsomhet"),
    ("føling", "hypoglykemi", "lavt blodsukker"),
    ("hyperglykemi", "høyt blodsukker"),
    ("fedme", "obesitas", "adipositas"),
    ("sykelig overvekt", "morbide obesitas", "alvorlig fedme"),
    ("undervekt", "lav kroppsvekt"),
    ("stoffskifte", "metabolisme"),
    ("lavt stoffskifte", "hypotyreose"),
    ("høyt stoffskifte", "hypertyreose"),
    ("høyt stoffskifte med giftstruma", "gravs sykdom", "graves sykdom"),
    ("hashimotos sykdom", "hashimoto tyreoiditt", "autoimmun tyreoiditt"),
    ("stoffskiftesykdom", "thyreoideasykdom"),
    ("metabolsk syndrom", "syndrom x"),
    ("urinsyregikt", "podagra", "gikt"),
    ("dehydrering", "uttørking"),
    ("væskemangel", "dehydrering"),
    ("elektrolyttforstyrrelse", "saltforstyrrelse"),
    ("for lavt natrium", "hyponatremi"),
    ("for høyt natrium", "hypernatremi"),
    ("for lavt kalium", "hypokalemi"),
    ("for høyt kalium", "hyperkalemi"),

    # ── Kreft og onkologi ─────────────────────────────────────────
    ("kreft", "cancer", "malign sykdom", "ondartet sykdom", "malignitet"),
    ("svulst", "tumor"),
    ("godartet svulst", "benign tumor"),
    ("ondartet svulst", "malign tumor"),
    ("brystkreft", "mammacancer", "mammakarsinom"),
    ("lungekreft", "lungecancer", "bronkialkreft"),
    ("prostatakreft", "prostatacancer"),
    ("hudkreft", "skin cancer"),
    ("melanom", "malignt melanom"),
    ("basalcellekreft", "basalcellekarsinom"),
    ("plateepitelkreft i hud", "plateepitelkarsinom i hud"),
    ("tarmkreft", "kolorektalkreft", "kolorektal kreft"),
    ("tykktarmskreft", "colonkreft", "kolonkreft"),
    ("endetarmskreft", "rektumkreft"),
    ("blærekreft", "urinblærekreft"),
    ("livmorhalskreft", "cervixcancer", "cervixkreft"),
    ("livmorkreft", "endometriekreft", "endometriecancer"),
    ("eggstokkreft", "ovarialkreft", "ovariecancer"),
    ("testikkelkreft", "testiscancer", "testikkelcancer"),
    ("nyrekreft", "nyrecellekarsinom", "renal cancer"),
    ("bukspyttkjertelkreft", "pankreaskreft", "pancreascancer"),
    ("spiserørskreft", "øsofaguskreft", "esofaguscancer"),
    ("magekreft", "ventrikkelkreft", "gastrisk cancer"),
    ("leverkreft", "hepatocellulært karsinom", "hcc"),
    ("galleveiskreft", "cholangiokarsinom", "kolangiokarsinom"),
    ("hjernesvulst", "hjernetumor"),
    ("glioblastom", "glioblastoma multiforme", "gbm"),
    ("leukemi", "blodkreft"),
    ("akutt myelogen leukemi", "aml"),
    ("akutt lymfatisk leukemi", "all"),
    ("kronisk lymfatisk leukemi", "cll"),
    ("kronisk myelogen leukemi", "cml"),
    ("lymfom", "lymfekreft"),
    ("hodgkins lymfom", "hodgkin lymfom"),
    ("non-hodgkins lymfom", "non hodgkin lymfom", "nhl"),
    ("myelomatose", "multippelt myelom"),
    ("metastase", "metastaser", "fjernspredning", "spredning"),
    ("strålebehandling", "radioterapi"),
    ("kjemoterapi", "cytostatika", "cellegift"),
    ("immunterapi", "checkpointhemmer-behandling"),
    ("målrettet behandling", "targeted therapy"),
    ("hormonbehandling ved kreft", "endokrin behandling"),
    ("remisjon", "sykdomsro"),
    ("tilbakefall", "residiv", "recidiv"),

    # ── Psykisk helse ─────────────────────────────────────────────
    ("depresjon", "depressiv lidelse", "major depresjon", "major depressive disorder"),
    ("vinterdepresjon", "sesongavhengig depresjon", "sad"),
    ("nedstemthet", "senket stemningsleie"),
    ("angst", "angstlidelse", "angstforstyrrelse"),
    ("generalisert angstlidelse", "gad"),
    ("panikklidelse", "panic disorder"),
    ("panikkanfall", "angstanfall"),
    ("sosial angst", "sosial fobi"),
    ("helseangst", "hypokondri", "hypokondrisk lidelse"),
    ("fobi", "fobisk lidelse"),
    ("schizofreni", "schizofren lidelse"),
    ("psykose", "psykotisk episode"),
    ("bipolar lidelse", "manisk depressiv lidelse", "bipolar affektiv lidelse"),
    ("mani", "manisk episode"),
    ("hypomani", "hypoman episode"),
    ("ptsd", "posttraumatisk stresslidelse", "posttraumatisk stressforstyrrelse"),
    ("kompleks ptsd", "c-ptsd", "kompleks posttraumatisk stresslidelse"),
    ("ocd", "tvangslidelse", "obsessiv kompulsiv lidelse"),
    ("adhd", "attention deficit hyperactivity disorder"),
    ("add", "attention deficit disorder"),
    ("autisme", "autismespekterforstyrrelse", "asd"),
    ("asperger", "asperger syndrom"),
    ("anoreksi", "anorexia nervosa"),
    ("bulimi", "bulimia nervosa"),
    ("overspisingslidelse", "binge eating disorder", "bed"),
    ("spiseforstyrrelse", "spiseforstyrrelser"),
    ("selvskading", "selvmutilering"),
    ("selvmord", "suicid"),
    ("selvmordstanker", "suicidale tanker"),
    ("søvnløshet", "insomni"),
    ("ruslidelse", "rusmiddelavhengighet", "substansbrukslidelse"),
    ("alkoholisme", "alkoholavhengighet", "alkoholbrukslidelse"),
    ("utbrenthet", "burnout"),
    ("personlighetsforstyrrelse", "personlighetslidelse"),
    ("emosjonelt ustabil personlighetsforstyrrelse", "borderline personlighetsforstyrrelse", "eupf", "bpd"),
    ("tvangspreget personlighetsforstyrrelse", "obsessiv-kompulsiv personlighetsforstyrrelse"),
    ("dissosiasjon", "dissosiative symptomer"),

    # ── Lunge og luftveier ────────────────────────────────────────
    ("kols", "kronisk obstruktiv lungesykdom", "copd"),
    ("astma", "bronkialastma"),
    ("lungebetennelse", "pneumoni"),
    ("luftveisinfeksjon", "respirasjonsinfeksjon"),
    ("forkjølelse", "øvre luftveisinfeksjon", "common cold"),
    ("bronkitt", "bronkialkatarr"),
    ("kronisk bronkitt", "chronic bronchitis"),
    ("bihulebetennelse", "sinusitt", "sinusitis"),
    ("halsbetennelse", "faryngitt"),
    ("mandelbetennelse", "tonsillitt"),
    ("strupebetennelse", "laryngitt"),
    ("tuberkulose", "tb", "tbc"),
    ("søvnapné", "obstruktiv søvnapné", "osa"),
    ("pustestans under søvn", "søvnapné"),
    ("pustevansker", "dyspné", "åndenød"),
    ("hyperventilering", "overpusting"),
    ("oksygenmangel", "hypoksi"),
    ("respirasjonssvikt", "lungesvikt"),
    ("lungefibrose", "pulmonal fibrose"),
    ("pneumothorax", "punktert lunge", "sammenfalt lunge"),
    ("pleuravæske", "pleuraeffusjon"),
    ("falsk krupp", "laryngitt hos barn", "pseudokrupp"),

    # ── Nevrologi ─────────────────────────────────────────────────
    ("demens", "demenssykdom"),
    ("alzheimers sykdom", "alzheimer"),
    ("vaskulær demens", "karsykdomsdemens"),
    ("frontotemporal demens", "ftd"),
    ("lewy body demens", "lewylegemedemens"),
    ("parkinsons sykdom", "parkinson"),
    ("epilepsi", "epileptisk sykdom"),
    ("epileptisk anfall", "krampeanfall"),
    ("migrene", "migreneanfall"),
    ("spenningshodepine", "tensjonshodepine"),
    ("clusterhodepine", "hortons hodepine"),
    ("multippel sklerose", "ms"),
    ("cerebral parese", "cp"),
    ("nevropati", "nerveskade"),
    ("perifer nevropati", "polynevropati"),
    ("hjernerystelse", "commotio cerebri", "commotio"),
    ("hjerneblødning", "intracerebral blødning", "intrakraniell blødning"),
    ("hjernehinnebetennelse", "meningitt"),
    ("hjernebetennelse", "encefalitt"),
    ("nervelidelse", "nevrologisk sykdom"),
    ("tics", "tic-lidelse"),
    ("tourette", "tourette syndrom"),
    ("essensiell tremor", "skjelving"),
    ("ansiktslammelse", "facialisparese", "bell parese", "bells parese"),
    ("nummenhet", "parestesi"),
    ("svimmelhet", "vertigo"),
    ("balanseforstyrrelse", "ustøhet"),
    ("delirium", "akutt forvirring"),
    ("rastløse bein", "restless legs", "rls", "urolige bein"),
    ("carpaltunnelsyndrom", "karpaltunnelsyndrom", "cts"),

    # ── Mage og tarm ──────────────────────────────────────────────
    ("forstoppelse", "obstipasjon"),
    ("diaré", "diare"),
    ("magesjau", "omgangssyke", "gastroenteritt"),
    ("irritabel tarm", "ibs", "irritabel tarmsyndrom"),
    ("crohns sykdom", "morbus crohn"),
    ("ulcerøs kolitt", "ulcerative colitis"),
    ("magesår", "ulcus", "peptisk ulcus"),
    ("tolvfingertarmsår", "duodenalsår", "duodenalulcus"),
    ("refluks", "reflukssykdom", "gastroøsofageal refluks", "gerd"),
    ("halsbrann", "pyrose"),
    ("sure oppstøt", "syreoppstøt", "refluks"),
    ("hepatitt", "leverbetennelse"),
    ("fettlever", "steatose", "hepatisk steatose"),
    ("levercirrhose", "skrumplever", "levercirrhose"),
    ("gulsott", "ikterus"),
    ("gallestein", "kolelithiasis"),
    ("galleblærebetennelse", "kolecystitt"),
    ("bukspyttkjertelbetennelse", "pankreatitt"),
    ("cøliaki", "coeliaki", "glutenintoleranse"),
    ("blindtarmbetennelse", "appendisitt"),
    ("hemoroider", "endetarmshemoroider"),
    ("endetarmsfissur", "analfissur"),
    ("tarmobstruksjon", "ileus", "tarmslyng"),
    ("oppblåst mage", "abdominal distensjon"),
    ("oppkast", "emesis"),
    ("kvalme", "nausea"),
    ("magesmerter", "abdominalsmerter", "abdominale smerter"),
    ("tarminvaginasjon", "invaginasjon"),
    ("brokk", "hernie"),
    ("lyske­brokk", "inguinalhernie"),
    ("navlebrokk", "umbilikalhernie"),
    ("endetarmsbetennelse", "proktitt"),
    ("helicobacter pylori", "h pylori"),

    # ── Nyre og urinveier ─────────────────────────────────────────
    ("nyresvikt", "nyreinsuffisiens"),
    ("akutt nyresvikt", "aki", "acute kidney injury"),
    ("kronisk nyresykdom", "ckd", "chronic kidney disease"),
    ("dialyse", "nyredialyse"),
    ("hemodialyse", "bloddialyse"),
    ("peritonealdialyse", "bukdialyse"),
    ("nyrestein", "urolithiasis", "nyresteiner"),
    ("urinveisinfeksjon", "uvi"),
    ("blærebetennelse", "cystitt"),
    ("nyrebekkenbetennelse", "pyelonefritt"),
    ("blod i urinen", "hematuri"),
    ("protein i urinen", "proteinuri"),
    ("urinretensjon", "urinopphopning"),
    ("inkontinens", "urininkontinens", "lekkasje"),
    ("stressinkontinens", "anstrengelsesinkontinens"),
    ("overaktiv blære", "oab"),
    ("sengevæting", "enurese"),
    ("smerter ved vannlating", "dysuri"),

    # ── Muskel og skjelett ────────────────────────────────────────
    ("artrose", "slitasjegikt"),
    ("revmatoid artritt", "leddgikt", "ra"),
    ("artritt", "ledd­betennelse"),
    ("bekhterevs sykdom", "ankyloserende spondylitt", "aksial spondyloartritt"),
    ("psoriasisartritt", "psoriasisgikt"),
    ("urinsyregikt", "gikt", "podagra"),
    ("osteoporose", "benskjørhet"),
    ("osteopeni", "lav benmasse"),
    ("ryggprolaps", "diskusprolaps", "skiveprolaps"),
    ("nakkeprolaps", "cervikal prolaps"),
    ("isjias", "lumbal radikulopati"),
    ("frossen skulder", "adhesiv kapsulitt"),
    ("senebetennelse", "tendinitt"),
    ("seneskjedebetennelse", "tenosynovitt"),
    ("slimposebetennelse", "bursitt"),
    ("fibromyalgi", "fibromyalgisyndrom"),
    ("beinskjørhet", "benskjørhet", "osteoporose"),
    ("brudd", "fraktur"),
    ("tretthetsbrudd", "stressfraktur"),
    ("hoftebrudd", "lårhalsbrudd"),
    ("håndleddsbrudd", "distal radiusfraktur"),
    ("ankelbrudd", "malleolfraktur"),
    ("forstuing", "distorsjon"),
    ("forstuet ankel", "ankeldistorsjon"),
    ("ledd ut av ledd", "luksasjon"),
    ("delvis ut av ledd", "subluksasjon"),
    ("skjev rygg", "skoliose"),
    ("krum rygg", "kyfose"),
    ("svai rygg", "lordose"),
    ("nakkevondt", "nakkesmerter", "cervikalgi"),
    ("ryggvondt", "ryggsmerter", "lumbago"),
    ("hekseskudd", "akutt lumbago"),
    ("bekkensmerter", "bekkenleddssmerter"),
    ("plantar fascitt", "plantarfasciitt", "hælspore"),
    ("musearm", "belastningsskade i arm"),
    ("golfalbue", "medial epikondylitt"),
    ("tennisalbue", "lateral epikondylitt"),

    # ── Hud ───────────────────────────────────────────────────────
    ("eksem", "dermatitt"),
    ("atopisk eksem", "atopisk dermatitt"),
    ("kontakteksem", "kontaktdermatitt"),
    ("seboreisk eksem", "seboreisk dermatitt"),
    ("psoriasis", "psoriasis vulgaris"),
    ("akne", "acne", "kviser"),
    ("rosacea", "rosacea-lidelse"),
    ("elveblest", "urticaria"),
    ("utslett", "eksantem"),
    ("hudinfeksjon", "kutan infeksjon"),
    ("cellulitt", "hudinfeksjon i underhuden"),
    ("erysipelas", "rosen"),
    ("impetigo", "brennkopper"),
    ("soppinfeksjon i hud", "hudsopp", "dermatofytose"),
    ("ringorm", "tinea corporis"),
    ("fotsopp", "tinea pedis"),
    ("neglesopp", "onychomykose"),
    ("skabb", "scabies"),
    ("lus", "hodelus", "pedikulose"),
    ("vorter", "verruca", "verrucae"),
    ("føflekk", "nevus"),
    ("byll", "abscess"),
    ("kokskade", "forbrenning"),
    ("solforbrenning", "erythema solare"),
    ("tørr hud", "xerose"),
    ("trykksår", "liggesår", "dekubitus"),
    ("helvetesild", "herpes zoster"),
    ("forkjølelsessår", "munnsår", "herpes simplex"),

    # ── Infeksjonssykdommer ───────────────────────────────────────
    ("infeksjon", "smittsom sykdom"),
    ("virusinfeksjon", "viral infeksjon"),
    ("bakterieinfeksjon", "bakteriell infeksjon"),
    ("soppinfeksjon", "mykose"),
    ("parasittinfeksjon", "parasitose"),
    ("influensa", "sesonginfluensa", "flu"),
    ("covid", "covid-19", "koronavirusinfeksjon", "sars-cov-2", "korona"),
    ("forkjølelse", "common cold"),
    ("rs-virus", "rsv", "respiratorisk syncytialvirus"),
    ("hiv", "humant immunsviktvirus"),
    ("aids", "ervervet immunsviktsyndrom"),
    ("klamydia", "klamydiainfeksjon"),
    ("gonoré", "gonore"),
    ("syfilis", "lues"),
    ("genital herpes", "herpes genitalis"),
    ("hpv", "humant papillomvirus"),
    ("borreliose", "lyme borreliosis", "lyme sykdom"),
    ("flåttencefalitt", "tbe", "tick-borne encephalitis"),
    ("mrsa", "meticillinresistent stafylokokk", "meticillinresistente stafylokokker"),
    ("sepsis", "blodforgiftning"),
    ("meningitt", "hjernehinnebetennelse"),
    ("encefalitt", "hjernebetennelse"),
    ("meslinger", "morbilli"),
    ("vannkopper", "varicella"),
    ("kikhoste", "pertussis"),
    ("kusma", "parotitt"),
    ("røde hunder", "rubella"),
    ("norovirus", "omgangssyke"),
    ("campylobacterinfeksjon", "campylobacteriose"),
    ("salmonella", "salmonellose"),
    ("shigellose", "shigella-infeksjon"),
    ("mononukleose", "kyssesyke", "epstein-barr-virusinfeksjon"),
    ("cytomegalovirus", "cmv"),
    ("hånd-fot-munn-sykdom", "hfmd"),
    ("ørebetennelse", "otitt"),
    ("mellomørebetennelse", "otitis media"),
    ("øyebetennelse", "konjunktivitt"),
    ("urinveisinfeksjon", "uvi"),
    ("blindtarmbetennelse", "appendisitt"),
    ("helikobakterinfeksjon", "helicobacter pylori-infeksjon"),

    # ── Kvinnehelse og svangerskap ────────────────────────────────
    ("graviditet", "svangerskap", "gravid", "svanger", "gravide"),
    ("fødsel", "forløsning", "nedkomst"),
    ("vaginal fødsel", "naturlig fødsel"),
    ("keisersnitt", "sectio", "sectio caesarea"),
    ("svangerskapsforgiftning", "preeklampsi"),
    ("eklampsi", "svangerskapskramper"),
    ("svangerskapsdiabetes", "gestasjonell diabetes"),
    ("utenomkvedsgraviditet", "ektopisk graviditet"),
    ("abort", "svangerskapsavbrudd"),
    ("spontanabort", "miscarriage"),
    ("missed abortion", "tilbakeholdt abort"),
    ("menstruasjon", "mensen"),
    ("menstruasjonssmerter", "dysmenoré"),
    ("uteblitt menstruasjon", "amenoré"),
    ("kraftig menstruasjon", "menoragi"),
    ("mellomblødning", "metroragi"),
    ("endometriose",),
    ("adenomyose",),
    ("pcos", "polycystisk ovariesyndrom"),
    ("infertilitet", "ufrivillig barnløshet"),
    ("overgangsalder", "menopause", "klimakteriet"),
    ("hetetokter", "varmebyger"),
    ("bekkenløsning", "bekkenleddssmerter i svangerskap"),
    ("bekkenbunnstrening", "bekkenbunnsøvelser", "bekkenbunn", "bekkenbunnsmuskeltrening"),
    ("framfall", "prolaps", "underlivsprolaps"),
    ("livmorframfall", "uterin prolaps"),
    ("vaginal tørrhet", "atrofisk vaginitt"),
    ("brystbetennelse", "mastitt"),
    ("amming", "brysternæring"),
    ("morsmelk", "brystmelk"),

    # ── Mannshelse ────────────────────────────────────────────────
    ("forstørret prostata", "benign prostatahyperplasi", "bph"),
    ("prostatabetennelse", "prostatitt"),
    ("erektil dysfunksjon", "impotens"),
    ("testikkelvridning", "torsio testis"),
    ("bitestikkelbetennelse", "epididymitt"),
    ("forhudsforsnevring", "fimose"),
    ("krum penis", "peyronies sykdom"),

    # ── Barn og ungdom ────────────────────────────────────────────
    ("kolikk", "spedbarnskolikk"),
    ("gulsott hos nyfødt", "nyfødtgulsott", "neonatal ikterus"),
    ("prematur", "for tidlig født"),
    ("plutselig spedbarnsdød", "sids"),
    ("vannhode", "hydrocefalus"),
    ("dysleksi", "lese- og skrivevansker"),
    ("språkvansker", "språkforstyrrelse"),
    ("utviklingshemming", "intellektuell funksjonsnedsettelse"),
    ("cerebral parese", "cp"),
    ("autisme", "autismespekterforstyrrelse"),
    ("adhd",),
    ("barnevaksinasjon", "vaksinasjon i barnevaksinasjonsprogrammet"),
    ("vaksine", "immunisering", "vaksinasjon"),

    # ── Øye ───────────────────────────────────────────────────────
    ("grå stær", "katarakt"),
    ("grønn stær", "glaukom"),
    ("aldersrelatert makuladegenerasjon", "amd"),
    ("netthinneløsning", "amotio retinae"),
    ("nærsynthet", "myopi"),
    ("langsynthet", "hyperopi"),
    ("skjeve hornhinner", "astigmatisme"),
    ("skjeling", "strabisme"),
    ("øyebetennelse", "konjunktivitt"),
    ("tørre øyne", "keratoconjunctivitis sicca"),
    ("grå stær operasjon", "kataraktoperasjon"),
    ("rødt øye", "okulær hyperemi"),

    # ── Øre, nese og hals ─────────────────────────────────────────
    ("ørebetennelse", "otitt"),
    ("mellomørebetennelse", "otitis media"),
    ("ytre ørebetennelse", "otitis externa", "svømmeøre"),
    ("øresus", "tinnitus"),
    ("hørselstap", "nedsatt hørsel"),
    ("tunghørthet", "hørselstap"),
    ("svimmelhetssykdom", "ménières sykdom", "menieres sykdom"),
    ("ørevokspropp", "cerumenpropp"),
    ("bihulebetennelse", "sinusitt"),
    ("neseblødning", "epistaksis"),
    ("allergisk rhinitt", "pollenallergi", "høysnue"),
    ("skjev neseskillevegg", "septumdeviasjon"),
    ("snorking", "ronkopati"),
    ("mandelbetennelse", "tonsillitt"),
    ("heshet", "dysfoni"),
    ("svelgevansker", "dysfagi"),
    ("munntørrhet", "xerostomi"),

    # ── Tannhelse og munn ─────────────────────────────────────────
    ("karies", "tannråte"),
    ("hull i tennene", "karies"),
    ("tannkjøttbetennelse", "gingivitt"),
    ("periodontitt", "tannkjøttsykdom"),
    ("tannløsning", "periodontitt"),
    ("visdomstannbetennelse", "perikoronitt"),
    ("munnsår", "aftøse sår", "afte"),
    ("tanngnissing", "bruksisme", "bruxisme"),
    ("kjeveleddssmerter", "tmd", "temporomandibulær dysfunksjon"),
    ("tannverk", "odontalgi"),

    # ── Allergi og immunologi ─────────────────────────────────────
    ("allergi", "allergisk reaksjon"),
    ("anafylaksi", "anafylaktisk reaksjon", "anafylaktisk sjokk"),
    ("pollenallergi", "allergisk rhinitt", "høysnue"),
    ("matallergi", "fødemiddelallergi"),
    ("peanøttallergi", "jordnøttallergi"),
    ("melkeallergi", "kumelkallergi"),
    ("eggeallergi", "allergi mot egg"),
    ("støvallergi", "husstøvmiddallergi"),
    ("dyrehårsallergi", "pelsdyrallergi"),
    ("insektstikkallergi", "vepsestikkallergi", "bistikkallergi"),
    ("overfølsomhet", "hypersensitivitet"),
    ("laktoseintoleranse", "melkesukkerintoleranse"),
    ("histaminintoleranse",),
    ("immunsvikt", "immunodefekt"),
    ("autoimmun sykdom", "autoimmun lidelse"),

    # ── Endokrinologi ─────────────────────────────────────────────
    ("binyresvikt", "adrenal insuffisiens"),
    ("addisons sykdom", "adrenokortikal insuffisiens"),
    ("cushings syndrom", "hyperkortisolisme"),
    ("hyperparatyreoidisme", "overfunksjon i biskjoldbruskkjertlene"),
    ("hypoparatyreoidisme", "underfunksjon i biskjoldbruskkjertlene"),
    ("struma", "forstørret skjoldbruskkjertel"),
    ("giftstruma", "toksisk struma"),

    # ── Blod og immunologi ────────────────────────────────────────
    ("anemi", "blodmangel"),
    ("jernmangelanemi", "anemi på grunn av jernmangel"),
    ("jernmangel", "lav jernstatus"),
    ("blødersykdom", "koagulasjonsforstyrrelse"),
    ("hemofili", "blødersykdom hemofili"),
    ("von willebrands sykdom", "vwd"),
    ("lav blodprosent", "anemi"),
    ("for få blodplater", "trombocytopeni"),
    ("for mange blodplater", "trombocytose"),
    ("lavt antall hvite blodceller", "leukopeni"),
    ("neutropeni", "lavt nøytrofiltall"),
    ("lymfeknutesvulst", "lymfom"),

    # ── Seksuell helse og kjønnssykdommer ─────────────────────────
    ("seksuelt overførbar infeksjon", "soi", "sti"),
    ("kjønnssykdom", "seksuelt overførbar sykdom"),
    ("klamydia", "chlamydia trachomatis-infeksjon"),
    ("gonoré", "gonore"),
    ("syfilis", "lues"),
    ("hpv", "humant papillomvirus"),
    ("kjønnsvorter", "kondylomer", "condyloma"),
    ("genital herpes", "herpes genitalis"),
    ("trikomonas", "trikomoniasis"),
    ("hiv",),
    ("hepatitt b", "hbv"),
    ("hepatitt c", "hcv"),

    # ── Kirurgi, undersøkelser og prosedyrer ──────────────────────
    ("operasjon", "kirurgi", "kirurgisk inngrep"),
    ("bedøvelse", "anestesi"),
    ("narkose", "generell anestesi"),
    ("lokalbedøvelse", "lokalanestesi"),
    ("vevsprøve", "biopsi"),
    ("blodprøve", "venøs prøve"),
    ("ultralyd", "sonografi"),
    ("røntgen", "radiografi"),
    ("ct", "ct-undersøkelse", "computertomografi"),
    ("mr", "mri", "magnetisk resonans"),
    ("koloskopi", "tykktarmsundersøkelse"),
    ("gastroskopi", "øvre endoskopi", "magespeiling"),
    ("sigmoidoskopi", "undersøkelse av endetarm og nedre tarm"),
    ("bronkoskopi", "luftveisspeiling"),
    ("cystoskopi", "blærespeiling"),
    ("mammografi", "brystundersøkelse med røntgen"),
    ("screening", "masseundersøkelse"),
    ("gjenoppliving", "hjerte-lunge-redning", "hlr", "cpr"),
    ("intubasjon", "luftveisrør"),
    ("stomi", "kunstig tarmåpning"),
    ("blodtransfusjon", "transfusjon"),

    # ── Legemidler og behandling ──────────────────────────────────
    ("legemiddel", "medisin", "medikament"),
    ("smertestillende", "analgetika"),
    ("paracetamol", "acetaminophen"),
    ("ibuprofen", "nsaid"),
    ("betennelsesdempende", "antiinflammatorisk"),
    ("antibiotika", "antibakterielle legemidler"),
    ("antidepressiva", "antidepressive legemidler"),
    ("ssri", "selektiv serotoninreopptakshemmer"),
    ("snri", "serotonin noradrenalin reopptakshemmer"),
    ("antipsykotika", "nevroleptika"),
    ("angstdempende", "anxiolytika"),
    ("sovemedisin", "hypnotika"),
    ("beroligende", "sedativa"),
    ("insulin", "insulinbehandling"),
    ("kortison", "kortikosteroider", "steroider"),
    ("prednisolon", "glukokortikoid"),
    ("cellegift", "kjemoterapi", "cytostatika"),
    ("immunterapi",),
    ("vaksine", "immunisering", "vaksinasjon"),
    ("blodfortynnende", "antikoagulantia"),
    ("blodtrykksmedisin", "antihypertensiva"),
    ("vanndrivende", "diuretika"),
    ("kolesterolsenkende", "lipidsenkende legemidler", "statiner"),
    ("magebeskyttende", "protonpumpehemmer", "ppi"),
    ("rehabilitering", "opptrening"),
    ("fysioterapi", "fysikalsk behandling"),
    ("ergoterapi", "ergoterapeutisk behandling"),
    ("palliativ behandling", "lindrende behandling", "palliasjon"),

    # ── Generelle symptomer og kliniske termer ────────────────────
    ("smerte", "smerter"),
    ("vondt", "smerte"),
    ("akutt", "plutselig oppstått"),
    ("kronisk", "langvarig", "vedvarende"),
    ("betennelse", "inflammasjon"),
    ("hevelse", "ødem", "opphovning"),
    ("blødning", "hemoragi"),
    ("feber", "forhøyet kroppstemperatur"),
    ("frysninger", "frostanfall"),
    ("slapphet", "kraftløshet"),
    ("utmattelse", "fatigue", "tretthet"),
    ("svimmelhet", "vertigo"),
    ("ørhet", "presynkope"),
    ("besvimelse", "synkope"),
    ("kvalme", "nausea"),
    ("oppkast", "emesis"),
    ("diaré", "løs mage"),
    ("forstoppelse", "treg mage"),
    ("pustevansker", "dyspné", "åndenød"),
    ("hoste", "tussis"),
    ("slimhoste", "produktiv hoste"),
    ("tørrhoste", "ikke-produktiv hoste"),
    ("brystsmerter", "smerter i brystet"),
    ("magesmerter", "abdominale smerter"),
    ("hodepine", "cefalgi"),
    ("ryggsmerter", "ryggvondt"),
    ("leddverk", "artralgi"),
    ("muskelsmerter", "myalgi"),
    ("hudkløe", "pruritus"),
    ("nummenhet", "parestesi"),
    ("prikking", "parestesi"),
    ("stivhet", "rigiditet"),
    ("vekttap", "utilsiktet vekttap"),
    ("vektøkning", "økning i kroppsvekt"),
    ("appetittløshet", "nedsatt appetitt"),
    ("munntørrhet", "xerostomi"),
    ("dehydrering", "uttørking"),
    ("bevisstløshet", "tap av bevissthet"),
    ("forvirring", "konfusjon"),
    ("hukommelsessvikt", "minnesvikt"),
    ("kramper", "anfall"),
    ("gulsott", "ikterus"),

    # ── Helsetjeneste og organisering ─────────────────────────────
    ("fastlege", "allmennlege", "primærlege"),
    ("legevakt", "akuttlegevakt"),
    ("akuttmottak", "mottak for øyeblikkelig hjelp"),
    ("sykehus", "hospital"),
    ("spesialisthelsetjeneste", "sykehushelsetjeneste"),
    ("helsestasjon", "barne- og familiehelsestasjon"),
    ("helsesykepleier", "skolehelsetjeneste-sykepleier"),
    ("sykepleier", "pleier"),
    ("psykolog", "psykologspesialist"),
    ("psykiater", "spesialist i psykiatri"),
    ("henvisning", "rekvisisjon", "henvisningsbrev"),
    ("epikrise", "utskrivningsnotat"),
    ("journal", "pasientjournal"),
    ("helseopplysninger", "pasientopplysninger"),
    ("egenandel", "pasientbetaling"),
    ("frikort", "egenandelstak-frikort"),

    # ── Livsstil og forebygging ───────────────────────────────────
    ("røyking", "tobakksbruk"),
    ("røykeslutt", "slutte å røyke"),
    ("snus", "snusbruk"),
    ("nikotinavhengighet", "nikotinavhengig"),
    ("fysisk aktivitet", "mosjon", "trening", "øvelser"),
    ("stillesitting", "inaktivitet", "sedentær"),
    ("kosthold", "ernæring", "diett"),
    ("sunn mat", "helsevennlig kosthold"),
    ("søvnhygiene", "gode søvnvaner"),
    ("vaksinasjon", "immunisering"),
    ("screening", "forebyggende undersøkelse"),
    ("forebygging", "prevensjon", "profylakse"),
    ("folkehelse", "befolkningshelse"),
    ("vitamin d", "d-vitamin", "vitamin d3"),
    ("kalsium", "calcium"),
    ("kroppsmasseindeks", "bmi", "body mass index"),
]


def _build_lookup() -> Dict[str, FrozenSet[str]]:
    """Build bidirectional synonym lookup from groups."""
    lookup: Dict[str, FrozenSet[str]] = {}
    for group in _SYNONYM_GROUPS:
        members = frozenset(group)
        for term in group:
            if term in lookup:
                # Merge with existing group (term appears in multiple groups)
                merged = lookup[term] | members
                for t in merged:
                    lookup[t] = merged
            else:
                lookup[term] = members
    return lookup


# Pre-built lookup: term → frozenset of all synonyms (including itself)
SYNONYM_LOOKUP: Dict[str, FrozenSet[str]] = _build_lookup()


def _tokenize_term(term: str) -> List[str]:
    """Tokenize a synonym term using the same \\w+ pattern as BM25Search."""
    return re.findall(r"\w+", term.lower())


def _build_multi_word_index() -> Dict[str, List[Tuple[str, List[str]]]]:
    """Index multi-word synonyms by their first token for fast lookup.

    Uses \\w+ tokenization so hyphenated terms like "covid-19" (tokens:
    ["covid", "19"]) are correctly treated as multi-word and indexed
    under their first token ("covid").
    """
    index: Dict[str, List[Tuple[str, List[str]]]] = {}
    for term in SYNONYM_LOOKUP:
        tokens = _tokenize_term(term)
        if len(tokens) > 1:
            index.setdefault(tokens[0], []).append((term, tokens))
    return index


_MULTI_WORD_INDEX: Dict[str, List[Tuple[str, List[str]]]] = _build_multi_word_index()

# Weight for synonym terms relative to original query terms (1.0).
# Lower weight means synonyms help recall without dominating scoring.
SYNONYM_WEIGHT: float = 0.5


def expand_terms(terms: List[str]) -> Dict[str, float]:
    """
    Expand query terms with weighted synonyms for BM25 scoring.

    Returns dict of term -> weight. Original terms get weight 1.0,
    synonyms get SYNONYM_WEIGHT (0.5) so they boost recall without
    overpowering the original query intent.

    Only single-word synonyms are added as expansion terms. Multi-word
    synonyms (e.g. "akutt hjerteinfarkt") are not split into individual
    words to avoid false matches on common words like "type" or "akutt".
    """
    weights: Dict[str, float] = {}
    for t in terms:
        weights[t] = weights.get(t, 0.0) + 1.0

    # Track which term indices are covered by multi-word matches
    covered_indices: Set[int] = set()

    # Multi-word query matches: check if query terms contain a multi-word
    # synonym as a contiguous token subsequence
    for i, term in enumerate(terms):
        if i in covered_indices:
            continue
        for candidate, candidate_tokens in _MULTI_WORD_INDEX.get(term, []):
            n = len(candidate_tokens)
            if i + n <= len(terms) and terms[i:i + n] == candidate_tokens:
                # Check no overlap with already covered indices
                span_indices = set(range(i, i + n))
                if not span_indices & covered_indices:
                    covered_indices |= span_indices
                    for syn in SYNONYM_LOOKUP[candidate]:
                        syn_tokens = _tokenize_term(syn)
                        if syn != candidate and len(syn_tokens) == 1:
                            if syn_tokens[0] not in weights:
                                weights[syn_tokens[0]] = SYNONYM_WEIGHT

    # Single-word synonyms (skip words covered by multi-word match)
    for i, term in enumerate(terms):
        if i in covered_indices:
            continue
        if term in SYNONYM_LOOKUP:
            for syn in SYNONYM_LOOKUP[term]:
                syn_tokens = _tokenize_term(syn)
                if syn != term and len(syn_tokens) == 1:
                    if syn_tokens[0] not in weights:
                        weights[syn_tokens[0]] = SYNONYM_WEIGHT

    return weights
