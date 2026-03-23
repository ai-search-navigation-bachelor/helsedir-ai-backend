ALTER TABLE content
ADD COLUMN IF NOT EXISTS nki_indicator_id VARCHAR(32) NULL AFTER document_url;

SET @idx_exists := (
    SELECT COUNT(1)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'content'
      AND index_name = 'idx_nki_indicator_id'
);
SET @create_index_sql := IF(
    @idx_exists = 0,
    'CREATE INDEX idx_nki_indicator_id ON content (nki_indicator_id)',
    'SELECT 1'
);
PREPARE stmt FROM @create_index_sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
