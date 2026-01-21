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
CREATE TABLE IF NOT EXISTS content (
    id VARCHAR(100) PRIMARY KEY,
    tittel TEXT NOT NULL,
    tekst LONGTEXT,
    info_type VARCHAR(50),
    url TEXT,
    koder JSON,
    tags JSON,
    maalgruppe JSON,
    embedding BLOB,
    INDEX idx_info_type (info_type),
    FULLTEXT INDEX idx_tittel (tittel),
    FULLTEXT INDEX idx_tekst (tekst)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Search Logs Table
-- ============================================================
-- Tracks all search queries made by users
CREATE TABLE IF NOT EXISTS search_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    search_id VARCHAR(36) NOT NULL,
    query TEXT NOT NULL,
    role VARCHAR(50),
    results_count INT DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_search_id (search_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Click Logs Table
-- ============================================================
-- Tracks user clicks on search results
CREATE TABLE IF NOT EXISTS click_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    search_id VARCHAR(36) NOT NULL,
    content_id VARCHAR(100) NOT NULL,
    position INT,
    query TEXT,
    role VARCHAR(50),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_search_id (search_id),
    INDEX idx_content_id (content_id),
    INDEX idx_timestamp (timestamp),
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Search Results Shown Table
-- ============================================================
-- Tracks which results were shown for each search (for learning-to-rank)
CREATE TABLE IF NOT EXISTS search_results_shown (
    id INT AUTO_INCREMENT PRIMARY KEY,
    search_id VARCHAR(36) NOT NULL,
    content_id VARCHAR(100) NOT NULL,
    position INT NOT NULL,
    score FLOAT DEFAULT 0,
    INDEX idx_search_id (search_id),
    INDEX idx_content_id (content_id),
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Content Statistics Table
-- ============================================================
-- Aggregated statistics for content performance
CREATE TABLE IF NOT EXISTS content_stats (
    content_id VARCHAR(100) PRIMARY KEY,
    impressions INT DEFAULT 0,
    clicks INT DEFAULT 0,
    ctr FLOAT AS (CASE WHEN impressions > 0 THEN (clicks / impressions) ELSE 0 END) VIRTUAL,
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Verification Query
-- ============================================================
-- Run this to verify all tables were created
SHOW TABLES;
