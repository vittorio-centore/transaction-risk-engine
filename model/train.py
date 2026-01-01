# training script for fraud detection model
# extracts data from database, trains pytorch model, evaluates performance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import asyncio
from datetime import datetime
from sklearn.model_selection import train_test_split
from sqlalchemy import select
import sys
import os

# add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.models import Transaction, db
from service.features import compute_all_features, get_feature_names
from model.model import FraudMLP, count_parameters
from model.evaluate import evaluate_model, plot_metrics

async def load_training_data(limit: int = None):
    """
    load transactions from database and compute features
    
    args:
        limit: optional max number of transactions to load (for testing)
        
    returns:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        transaction_ids: list of transaction ids
    """
    
    print("📂 loading transactions from database...")
    
    async with db.async_session() as session:
        # get transactions with all needed fields
        query = select(Transaction).order_by(Transaction.timestamp)
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        transactions = result.scalars().all()
    
    print(f"✅ loaded {len(transactions)} transactions")
    
    # fraud distribution
    labels = np.array([tx.label for tx in transactions])
    fraud_count = labels.sum()
    fraud_ratio = fraud_count / len(labels)
    print(f"   fraud ratio: {fraud_ratio:.2%} ({fraud_count} fraud, {len(labels) - fraud_count} legit)")
    
    print("🔧 computing features for each transaction...")
    
    X_list = []
    y_list = []
    tx_ids = []
    
    for i, tx in enumerate(transactions):
        if i % 10000 == 0 and i > 0:
            print(f"   processed {i}/{len(transactions)} transactions...")
        
        # Compute features using new 18-feature pipeline (NO v_features!)
        try:
            feature_vector = await compute_all_features(
                user_id=tx.user_id,
                merchant_id=tx.merchant_id,
                amount=float(tx.amount),
                timestamp=tx.timestamp,
                exclude_tx_id=tx.transaction_id
            )
            
            X_list.append(feature_vector)
            y_list.append(tx.label)
            tx_ids.append(tx.transaction_id)
            
        except Exception as e:
            print(f"⚠️  error processing tx {tx.transaction_id}: {e}")
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ computed features for {len(X)} transactions")
    print(f"   feature matrix shape: {X.shape}")
    
    return X, y, tx_ids

def train_model(
    X_train, y_train,
    X_val, y_val,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001
):
    """
    train pytorch fraud detection model
    
    args:
        X_train, y_train: training data
        X_val, y_val: validation data
        epochs: max training epochs
        batch_size: batch size
        learning_rate: learning rate for adam optimizer
        
    returns:
        trained model
    """
    
    print("\n🎯 training fraud detection model...")
    
    # calculate class imbalance weight
    fraud_ratio = y_train.sum() / len(y_train)
    pos_weight = torch.tensor([(1 - fraud_ratio) / fraud_ratio])
    
    print(f"📊 class imbalance:")
    print(f"   fraud ratio: {fraud_ratio:.2%}")
    print(f"   pos_weight for loss: {pos_weight.item():.2f}")
    
    # initialize model
    input_dim = X_train.shape[1]
    model = FraudMLP(input_dim=input_dim)
    
    print(f"🏗️  model architecture:")
    print(f"   input dim: {input_dim}")
    print(f"   parameters: {count_parameters(model):,}")
    
    # loss function with class weighting (handles imbalance)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # validation tensors
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        print(f"epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/fraud_model_v1.pt')
            print(f"   ✅ saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⏹️  early stopping after {epoch+1} epochs")
                break
    
    # load best model
    checkpoint = torch.load('models/fraud_model_v1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

async def main():
    """main training pipeline"""
    
    print("🚀 fraud detection model training pipeline\n")
    
    # 1. load data (use subset for faster training during development)
    # TODO: change to None to use full dataset
    X, y, tx_ids = await load_training_data(limit=50000)
    
    # TIME-BASED SPLIT (production-realistic)
    # Train on first 60%, val on next 20%, test on final 20%
    # This matches real scenario: predict future fraud based on past data
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\n📊 dataset splits (TIME-BASED):")
    print(f"   train: {len(X_train)} ({y_train.sum()} fraud, {(y_train.sum()/len(y_train)*100):.2f}%)")
    print(f"   val:   {len(X_val)} ({y_val.sum()} fraud, {(y_val.sum()/len(y_val)*100):.2f}%)")
    print(f"   test:  {len(X_test)} ({y_test.sum()} fraud, {(y_test.sum()/len(y_test)*100):.2f}%)")
    
    # 2.5. CRITICAL: scale features (neural networks need this!)
    print("\n🔧 scaling features...")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   median: {np.median(X_train_scaled):.6f}")
    print(f"   IQR: {np.percentile(X_train_scaled, 75) - np.percentile(X_train_scaled, 25):.6f}")
    
    # save scaler for inference
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   ✅ saved scaler to models/scaler.pkl")
    
    # 3. train model with SCALED data
    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 4. evaluate on test set (SCALED)
    print("\n📈 evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    print("\n✅ training complete!")
    print(f"   model saved to: models/fraud_model_v1.pt")
    print(f"   scaler saved to: models/scaler.pkl")
    print(f"   test pr-auc: {metrics['pr_auc']:.4f}")
    print(f"   test precision: {metrics['precision']:.4f}")
    print(f"   test recall: {metrics['recall']:.4f}")
    
    # 5. generate evaluation plots
    print("\n📊 generating evaluation plots...")
    plot_metrics(model, X_test_scaled, y_test, save_path='models/evaluation_plots.png')
    
    print("\n🎉 all done! next: run the api with the trained model")

if __name__ == '__main__':
    asyncio.run(main())
