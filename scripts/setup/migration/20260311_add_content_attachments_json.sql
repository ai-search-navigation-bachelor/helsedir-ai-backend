ALTER TABLE content
    ADD COLUMN attachments_json JSON NULL AFTER document_url;
