-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1. USERS (Auth & Scoping)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. AGENT RUNS (Who, What, Answer, Cost, Status)
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_agent_runs_user_id ON agent_runs(user_id);
CREATE INDEX idx_agent_runs_created_at ON agent_runs(created_at);

-- 3. TOOL LOGS (Which tools fired, payloads, latency)
CREATE TABLE tool_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    input_payload JSONB,
    output_payload JSONB,
    status VARCHAR(50) DEFAULT 'success',
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_tool_logs_run_id ON tool_logs(run_id);

-- 4. DOCUMENTS (RAG Knowledge + pgvector)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
-- Fast cosine similarity search
CREATE INDEX idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops);
-- Optional: fast metadata filtering (e.g., WHERE metadata->>'style' = 'budget')
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);

-- 5. DELIVERY LOGS (Resend/Email audit trail)
CREATE TABLE delivery_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_email VARCHAR(255) NOT NULL,
    subject VARCHAR(255),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    provider_message_id VARCHAR(255),
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_delivery_logs_run_id ON delivery_logs(run_id);
CREATE INDEX idx_delivery_logs_user_id ON delivery_logs(user_id);