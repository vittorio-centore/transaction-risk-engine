-- transaction risk engine database schema
-- this creates all tables + performance indexes for fast time-window queries

-- users table: basic user info
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    home_country VARCHAR(3),  -- ISO country code like 'USA', 'RUS', etc
    account_age_days INT
);

-- merchants table: where transactions happen
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id SERIAL PRIMARY KEY,
    category VARCHAR(50),  -- e.g. 'electronics', 'groceries', 'travel'
    country VARCHAR(3),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- transactions table: the main event log
-- includes kaggle dataset's anonymized features (v1-v28 from pca)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    merchant_id INT REFERENCES merchants(merchant_id),
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    country VARCHAR(3),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    label INT NOT NULL,  -- 0 = legit, 1 = fraud
    
    -- anonymized features from kaggle credit card fraud dataset
    -- these are pca components (v1-v28) + time + amount
    v1 DECIMAL, v2 DECIMAL, v3 DECIMAL, v4 DECIMAL, v5 DECIMAL,
    v6 DECIMAL, v7 DECIMAL, v8 DECIMAL, v9 DECIMAL, v10 DECIMAL,
    v11 DECIMAL, v12 DECIMAL, v13 DECIMAL, v14 DECIMAL, v15 DECIMAL,
    v16 DECIMAL, v17 DECIMAL, v18 DECIMAL, v19 DECIMAL, v20 DECIMAL,
    v21 DECIMAL, v22 DECIMAL, v23 DECIMAL, v24 DECIMAL, v25 DECIMAL,
    v26 DECIMAL, v27 DECIMAL, v28 DECIMAL
);

-- risk_scores table: audit log of all predictions
CREATE TABLE IF NOT EXISTS risk_scores (
    id SERIAL PRIMARY KEY,
    transaction_id INT REFERENCES transactions(transaction_id),
    model_version VARCHAR(50),  -- track which model version scored this
    risk_score DECIMAL(5, 4),  -- 0.0000 to 1.0000 probability
    decision VARCHAR(10),  -- 'approve', 'review', or 'decline'
    top_features JSONB,  -- json array of top contributing features for explainability
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- CRITICAL PERFORMANCE INDEXES
-- these make "last N minutes/hours" queries lightning fast
-- without these, queries would scan entire table (millions of rows)

-- index for user velocity queries: "how many tx has user X made in last hour?"
-- covering user_id + timestamp makes these queries super fast
CREATE INDEX IF NOT EXISTS idx_tx_user_time ON transactions(user_id, timestamp DESC);

-- index for merchant baseline queries: "what's merchant Y's fraud rate in last 30 days?"
CREATE INDEX IF NOT EXISTS idx_tx_merchant_time ON transactions(merchant_id, timestamp DESC);

-- index for general time-based queries
CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(timestamp DESC);

-- index for label lookups during training/evaluation
CREATE INDEX IF NOT EXISTS idx_tx_label ON transactions(label);

-- create some sample users and merchants for testing
-- we'll generate more when loading the kaggle dataset
INSERT INTO users (home_country, account_age_days) 
SELECT 
    CASE (random() * 5)::int
        WHEN 0 THEN 'USA'
        WHEN 1 THEN 'GBR'
        WHEN 2 THEN 'CAN'
        WHEN 3 THEN 'DEU'
        ELSE 'FRA'
    END,
    (random() * 1000)::int
FROM generate_series(1, 100);

INSERT INTO merchants (category, country)
SELECT
    CASE (random() * 4)::int
        WHEN 0 THEN 'retail'
        WHEN 1 THEN 'travel'
        WHEN 2 THEN 'electronics'
        ELSE 'groceries'
    END,
    CASE (random() * 5)::int
        WHEN 0 THEN 'USA'
        WHEN 1 THEN 'GBR'
        WHEN 2 THEN 'CAN'
        WHEN 3 THEN 'DEU'
        ELSE 'FRA'
    END
FROM generate_series(1, 50);

-- add a comment explaining the schema
COMMENT ON TABLE transactions IS 'main transaction log with anonymized features from kaggle dataset';
COMMENT ON COLUMN transactions.label IS '0 = legitimate transaction, 1 = fraudulent transaction';
COMMENT ON INDEX idx_tx_user_time IS 'speeds up user velocity queries (tx count in last N hours)';
