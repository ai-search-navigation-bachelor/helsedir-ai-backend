"""
Services package — core application logic and data access.

Organized into sub-packages:
- repositories/  MySQL data access (Repository pattern)
- data/          Content loading, caching, and metadata
- search/        BM25, semantic, hybrid search, RRF fusion, ML reranking
- statistics/    NKI quality indicator fetching and caching
- analytics/     Event logging (searches, clicks, impressions)
- external/      HTTP client for the Helsedirektoratet API
"""
