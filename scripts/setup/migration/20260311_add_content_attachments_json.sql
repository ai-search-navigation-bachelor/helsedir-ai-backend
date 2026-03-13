ALTER TABLE content
    ADD COLUMN attachments_json JSON NULL AFTER document_url,
    ADD COLUMN ehelsestandard_fields_json JSON NULL AFTER attachments_json;
