-- ============================================================
-- Helsedir AI Backend - Database Initialization
-- ============================================================
-- Creates all required tables for the application
-- Run this script to set up a fresh database

-- Create database (if needed)
CREATE DATABASE IF NOT EXISTS helsedir_ai
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE helsedir_ai;

-- ============================================================
-- Content Table
-- ============================================================
-- Stores health content from Helsedirektoratet with embeddings
-- Also stores theme pages (info_type='temaside') for navigation hierarchy
CREATE TABLE IF NOT EXISTS content (
    id VARCHAR(100) PRIMARY KEY,
    tittel TEXT NOT NULL,
    tekst LONGTEXT,
    info_type VARCHAR(50),
    koder JSON,
    maalgruppe JSON,
    links JSON,
    embedding BLOB,

    -- Theme page fields (only used when info_type='temaside')
    -- Also stores helsedirektoratet.no path for regular content (from API url field)
    path VARCHAR(1000),

    INDEX idx_info_type (info_type),
    INDEX idx_path (path(255)),
    FULLTEXT INDEX idx_tittel (tittel),
    FULLTEXT INDEX idx_tekst (tekst)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Anbefaling Details Table
-- ============================================================
-- Stores anbefaling-specific fields from /innhold/anbefalinger/{id}
-- Separate table to avoid NULL-spam in content table (normalization)
CREATE TABLE IF NOT EXISTS anbefaling_details (
    content_id VARCHAR(255) PRIMARY KEY,
    praktisk LONGTEXT DEFAULT NULL,
    rasjonale LONGTEXT DEFAULT NULL,
    fordeler_ulemper LONGTEXT DEFAULT NULL,
    verdier_preferanser LONGTEXT DEFAULT NULL,
    kvalitet_dokumentasjon LONGTEXT DEFAULT NULL,
    ressurshensyn LONGTEXT DEFAULT NULL,
    styrke VARCHAR(100) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Foreign key to ensure referential integrity
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE,

    -- Index for potential future search on styrke
    INDEX idx_styrke (styrke)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Theme Page Content Junction Table
-- ============================================================
-- Links theme pages to their associated content items
-- Allows many-to-many relationship (one content can appear on multiple theme pages)
CREATE TABLE IF NOT EXISTS theme_page_content (
    id INT PRIMARY KEY AUTO_INCREMENT,
    theme_page_id VARCHAR(100) NOT NULL,
    content_id VARCHAR(100) NOT NULL,
    display_order INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (theme_page_id) REFERENCES content(id) ON DELETE CASCADE,
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE,
    INDEX idx_theme_page (theme_page_id),
    INDEX idx_content (content_id),
    UNIQUE KEY unique_theme_content (theme_page_id, content_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Search Logs Table
-- ============================================================
-- Tracks all search queries made by users
CREATE TABLE IF NOT EXISTS search_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    search_id VARCHAR(36) UNIQUE NOT NULL,
    query TEXT NOT NULL,
    role VARCHAR(100),
    session_id VARCHAR(36),
    user_id VARCHAR(36),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Search Results Shown Table
-- ============================================================
-- Tracks which results were shown for each search with ML features
CREATE TABLE IF NOT EXISTS search_results_shown (
    id INT PRIMARY KEY AUTO_INCREMENT,
    search_id VARCHAR(36) NOT NULL,
    content_id VARCHAR(100) NOT NULL,
    position INT NOT NULL,
    score FLOAT,

    -- Semantic signal
    semantic_similarity FLOAT,

    -- Keyword signals (RAW scores - normalized during training)
    keyword_score_total FLOAT,
    exact_title_proportion FLOAT,
    full_coverage_proportion FLOAT,
    title_keyword_proportion FLOAT,

    -- Metadata signals
    type_match FLOAT,
    role_match FLOAT,
    code_match_count INT DEFAULT 0,
    lis_match TINYINT DEFAULT 0,
    maalgruppe_match TINYINT DEFAULT 0,

    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (search_id) REFERENCES search_logs(search_id),
    INDEX idx_search_id (search_id),
    INDEX idx_content_id (content_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Click Logs Table
-- ============================================================
-- Tracks user clicks on search results
CREATE TABLE IF NOT EXISTS click_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    search_id VARCHAR(36) NOT NULL,
    content_id VARCHAR(100) NOT NULL,
    position INT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (search_id) REFERENCES search_logs(search_id),
    INDEX idx_search_id (search_id),
    INDEX idx_content_id (content_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Content Statistics Table
-- ============================================================
-- Aggregated statistics for content performance
CREATE TABLE IF NOT EXISTS content_stats (
    content_id VARCHAR(100) PRIMARY KEY,
    clicks INT DEFAULT 0,
    impressions INT DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Position Propensity Table
-- ============================================================
-- Stores learned position bias for IPS weighting in LTR training
CREATE TABLE IF NOT EXISTS position_propensity (
    position INT PRIMARY KEY,
    propensity FLOAT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default propensity values
INSERT INTO position_propensity (position, propensity) VALUES
(1, 1.00),
(2, 0.70),
(3, 0.55),
(4, 0.45),
(5, 0.40),
(6, 0.35),
(7, 0.30),
(8, 0.28),
(9, 0.26),
(10, 0.24)
ON DUPLICATE KEY UPDATE propensity = VALUES(propensity);

-- ============================================================
-- Verification Query
-- ============================================================
SHOW TABLES;
