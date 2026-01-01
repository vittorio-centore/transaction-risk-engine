# Transaction Risk Engine - Setup Guide

## For New Users Cloning This Repository

### Prerequisites
- Docker Desktop installed and running
- 4GB RAM available
- 5GB disk space
- Internet connection (for dataset download)

### Quick Start (5 Minutes)

#### 1. Clone Repository
```bash
git clone <your-repo-url>
cd transaction-risk
```

#### 2. Download Dataset
Download Sparkov Credit Card dataset:
- **Option A:** [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) 
- **Option B:** Direct link (if available)

Place in `sparkov_data/` directory:
```bash
mkdir -p sparkov_data
# Copy your sparkov CSV files here
```

#### 3. Deploy Everything
```bash
./scripts/deploy_docker.sh
```

This will:
- ✅ Build Docker images
- ✅ Start PostgreSQL + API
- ✅ Load 1.3M transactions
- ✅ Create indexes

#### 4. Train Model
```bash
docker-compose exec api python model/train.py
```

Takes ~3 minutes. Creates:
- `models/fraud_model_v1.pt` (trained model)
- `models/scaler.pkl` (feature scaler)
- `models/thresholds.json` (decision thresholds)

#### 5. Test API
```bash
# Pretty formatted
python tests/pretty_test.py

# Or raw curl
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "merchant_id": 1, "amount": 100.00}'
```

### What's NOT in Git (You Generate Locally)

**❌ Not Committed:**
- `models/*.pt` - Trained model (generate with training)
- `models/*.pkl` - Scalers (generate with training)
- `*.csv` - Datasets (too large, download separately)
- `.env` - Your local config

**✅ Committed:**
- All source code
- Docker configuration
- Documentation
- Empty `models/` directory structure

### Project Structure

```
transaction-risk/
├── db/                  # Database layer
├── service/             # API + features
├── model/               # Training scripts
├── tests/               # Testing tools
├── models/              # Generated (not in git!)
│   ├── .gitkeep        # ✅ Keeps directory
│   └── README.md       # ✅ Instructions
├── docker-compose.yml   # ✅ Deployment config
└── README.md           # ✅ This file
```

### Expected Performance

After training, you should see:
- **50% recall @ 1% FPR** - Catches half of fraud with 1% false positives
- **30% precision @ 1% review** - 45x better than random
- **API latency:** 50-100ms per request
- **Throughput:** ~100 req/sec

### Troubleshooting

**Model files missing?**
```bash
docker-compose exec api python model/train.py
```

**Database connection failed?**
```bash
docker-compose down
docker-compose up -d
```

**API not responding?**
```bash
docker-compose logs api
```

### Documentation

- `DOCKER_README.md` - Docker deployment guide
- `walkthrough.md` - Complete technical walkthrough
- `PROJECT_SUMMARY.md` - Project overview

### Next Steps

1. ✅ Setup complete
2. Train your model on your data
3. Customize thresholds for your business
4. Monitor performance
5. Iterate!

---

**This is production-ready fraud detection, not a tutorial!**
