ALTER TABLE content
    ADD COLUMN has_text_content TINYINT(1) NULL AFTER links,
    ADD COLUMN document_url TEXT NULL AFTER has_text_content;
