"""
Statistics services — NKI quality indicator fetching and matching.

nki_statistics_service.py  Fetches NKI indicators from Helsedirektoratet's
                           NKI API and caches results (default TTL: 6 hours).
nki_matching.py            Matches indicator IDs to content items by URL and
                           title normalization.
"""
