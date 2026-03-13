INSERT INTO content_type_config (info_type, searchable, display_name)
VALUES
    ('ehelsestandard', 1, 'E-helsestandard'),
    ('e-helsestandard', 1, 'E-helsestandard')
ON DUPLICATE KEY UPDATE
    searchable = VALUES(searchable),
    display_name = VALUES(display_name);
