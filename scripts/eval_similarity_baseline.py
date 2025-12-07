#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data(path: str):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        # Need to parse string representation of lists if loaded from CSV
        df = pd.read_csv(path)
        for col in ["anchor_features", "pos_features", "neg_features"]:
            df[col] = df[col].apply(eval)
        return df
    else:
        raise ValueError("Unsupported file format")

def evaluate_baseline(df: pd.DataFrame):
    print(f"Evaluating baseline on {len(df)} triplets...")
    
    anchors = np.stack(df["anchor_features"].values)
    positives = np.stack(df["pos_features"].values)
    negatives = np.stack(df["neg_features"].values)
    
    # Compute similarities
    # cosine_similarity returns matrix, we want diagonal
    sim_pos = np.sum(anchors * positives, axis=1) / (np.linalg.norm(anchors, axis=1) * np.linalg.norm(positives, axis=1))
    sim_neg = np.sum(anchors * negatives, axis=1) / (np.linalg.norm(anchors, axis=1) * np.linalg.norm(negatives, axis=1))
    
    # Metrics
    correct = np.sum(sim_pos > sim_neg)
    accuracy = correct / len(df)
    
    margin = sim_pos - sim_neg
    avg_margin = np.mean(margin)
    
    print(f"Baseline Results (Cosine Similarity on Raw Features):")
    print(f"  Triplet Accuracy: {accuracy:.4f} (sim(a,p) > sim(a,n))")
    print(f"  Average Margin:   {avg_margin:.4f}")
    print(f"  Avg Sim(Pos):     {np.mean(sim_pos):.4f}")
    print(f"  Avg Sim(Neg):     {np.mean(sim_neg):.4f}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate similarity baseline")
    parser.add_argument("--input", type=str, default="data/metric_learning/triplets.parquet", help="Input data path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found. Run export_metric_learning_data.py first.")
        sys.exit(1)
        
    df = load_data(args.input)
    evaluate_baseline(df)

if __name__ == "__main__":
    main()
