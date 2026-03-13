ALTER TABLE search_logs
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(36) NULL AFTER role,
    ADD COLUMN IF NOT EXISTS user_id VARCHAR(36) NULL AFTER session_id;

SET @idx_exists = (
    SELECT COUNT(*)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'search_logs'
      AND index_name = 'idx_session'
);
SET @create_idx_sql = IF(
    @idx_exists = 0,
    'CREATE INDEX idx_session ON search_logs (session_id)',
    'SELECT 1'
);
PREPARE create_idx_stmt FROM @create_idx_sql;
EXECUTE create_idx_stmt;
DEALLOCATE PREPARE create_idx_stmt;

ALTER TABLE search_results_shown
    ADD COLUMN IF NOT EXISTS bm25_score FLOAT NULL AFTER semantic_score,
    ADD COLUMN IF NOT EXISTS rrf_score FLOAT NULL AFTER bm25_score,
    ADD COLUMN IF NOT EXISTS type_match FLOAT NULL AFTER rrf_score,
    ADD COLUMN IF NOT EXISTS role_match FLOAT NULL AFTER type_match,
    ADD COLUMN IF NOT EXISTS maalgruppe_match TINYINT DEFAULT 0 AFTER role_match;

CREATE TABLE IF NOT EXISTS content_stats (
    content_id VARCHAR(100) PRIMARY KEY,
    clicks INT DEFAULT 0,
    impressions INT DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
