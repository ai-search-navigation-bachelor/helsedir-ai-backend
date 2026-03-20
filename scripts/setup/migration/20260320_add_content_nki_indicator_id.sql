ALTER TABLE content
ADD COLUMN IF NOT EXISTS nki_indicator_id VARCHAR(32) NULL AFTER document_url;

CREATE INDEX idx_nki_indicator_id ON content (nki_indicator_id);
