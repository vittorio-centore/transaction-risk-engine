# Transaction Risk Engine - Docker Quick Start

## Prerequisites

- Docker Desktop installed and running
- 4GB RAM available
- 5GB disk space

## One-Command Deployment

```bash
./scripts/deploy_docker.sh
```

This script will:
1. ✅ Build Docker images
2. ✅ Start PostgreSQL + API
3. ✅ Load Sparkov dataset (1.3M transactions)
4. ✅ Create database indexes
5. ✅ Verify model files

## Manual Deployment

### Step 1: Start Services

```bash
docker-compose up -d
```

### Step 2: Check Health

```bash
# Check all services running
docker-compose ps

# Check API health
curl http://localhost:8000/health
```

### Step 3: Load Data (First Time Only)

```bash
# Load Sparkov dataset
docker-compose exec api python db/load_sparkov.py

# Create indexes for performance
docker-compose exec api python db/add_indexes.py
```

### Step 4: Train Model (If Not Trained)

```bash
docker-compose exec api python model/train.py
```

## Testing the API

### Score a Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "merchant_id": 1,
    "amount": 100.00
  }'
```

**Expected Response:**
```json
{
  "fraud_score": 0.1234,
  "decision": "approve",
  "reason": "Low risk transaction",
  "top_features": [
    {"feature": "tx_count_24h", "value": 3.0},
    {"feature": "amount", "value": 100.0}
  ]
}
```

## Useful Commands

### View Logs
```bash
# API logs
docker-compose logs -f api

# Database logs
docker-compose logs -f postgres

# All logs
docker-compose logs -f
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U frauduser -d frauddb

# Run SQL queries
docker-compose exec postgres psql -U frauduser -d frauddb -c "SELECT COUNT(*) FROM transactions;"
```

### Stop/Restart
```bash
# Stop services (keeps data)
docker-compose down

# Stop and remove data
docker-compose down -v

# Restart specific service
docker-compose restart api
```

### Rebuild After Code Changes
```bash
# Rebuild API image
docker-compose build api

# Restart with new image
docker-compose up -d api
```

## Troubleshooting

### Port Already in Use
```bash
# If port 8000 or 5432 is busy
docker-compose down
# Change ports in docker-compose.yml:
# ports: - "8001:8000"  # Use 8001 instead
```

### API Not Starting
```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Model files missing → Run training
# 2. Database not ready → Wait 30s and retry
# 3. Port conflict → Change port in docker-compose.yml
```

### Data Not Loading
```bash
# Check if Sparkov CSV exists
ls sparkov_data/

# Re-download if needed
# (Instructions in main README)
```

### Model Scores All Zero
```bash
# This means model not trained properly
# Retrain with:
docker-compose exec api python model/train.py

# Check features are 19 (not 38):
docker-compose exec api python -c "from service.features import get_feature_names; print(len(get_feature_names()))"
# Should output: 19
```

## Production Deployment

### Environment Variables

Create `.env` file:
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
MODEL_PATH=/app/models/fraud_model_v1.pt
LOG_LEVEL=warning
```

### Security Checklist

- [ ] Change PostgreSQL password in docker-compose.yml
- [ ] Use environment variables (not hardcoded)
- [ ] Enable SSL for API (nginx reverse proxy)
- [ ] Limit database connections
- [ ] Set up monitoring (health checks)
- [ ] Regular backups of PostgreSQL volume

### Scaling

```bash
# Run 3 API replicas
docker-compose up -d --scale api=3

# Use nginx for load balancing
```

## Architecture

```
┌─────────────────┐
│   Docker Host   │
│                 │
│  ┌───────────┐  │
│  │  API:8000 │  │ ← FastAPI + PyTorch
│  │  (19 feat)│  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │PostgreSQL │  │ ← 1.3M transactions
│  │   :5432   │  │   Indexed for performance
│  └───────────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │  Volume   │  │ ← Persistent data
│  │ (survives)│  │
│  └───────────┘  │
└─────────────────┘
```

## Performance

- **API latency:** 50-100ms per score
- **Features:** All computed server-side from DB
- **Model:** 12.9k parameters, 19 features
- **Throughput:** ~100 req/sec (single container)

## Next Steps

1. Test API with sample transactions
2. Review business metrics: `cat models/thresholds.json`
3. Monitor performance: Check logs
4. Set up production environment with SSL
5. Configure monitoring/alerting

For detailed walkthrough: See `walkthrough.md` artifact
