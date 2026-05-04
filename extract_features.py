"""
Packet Extraction and Flow Feature Computation
===============================================
Processes Wireshark CSV exports to create 1-second window features.

This script:
1. Reads CSV exports from Wireshark (amazon_capture.csv, hulu_capture.csv, youtube_capture.csv)
2. Groups packets into 1-second time windows
3. Computes 9 flow-level features per window
4. Labels each window by source (amazon/hulu/youtube)
5. Outputs combined dataset for machine learning

Run:
    python extract_features.py
    
Outputs: combined_second_flows.csv
"""

import pandas as pd
import numpy as np
from collections import Counter

def extract_packets_from_csv(csv_path, label):
    """
    Read Wireshark CSV export and extract packet-level data.
    
    Args:
        csv_path: Path to Wireshark CSV export
        label: Class label (amazon, hulu, or youtube)
    
    Returns:
        DataFrame with columns: frame_num, time, length, label, second
    """
    df = pd.read_csv(csv_path)
    
    # Rename columns from Wireshark format
    df = df.rename(columns={
        'No.': 'frame_num',
        'Time': 'time',
        'Length': 'length'
    })
    
    # Convert to proper types
    df['time'] = df['time'].astype(float)
    df['length'] = df['length'].astype(int)
    df['label'] = label
    
    # Assign each packet to a 1-second window
    # floor(timestamp) groups all packets in the same second together
    df['second'] = df['time'].apply(lambda t: int(t))
    
    print(f"{label}: {len(df)} packets across {df['second'].max() - df['second'].min():.0f} seconds")
    
    return df


def compute_flow_features(packet_df):
    """
    Compute 9 flow-level features for each 1-second window.
    
    Features:
    1. Packet Count
    2. Total Length (bytes)
    3. Average Packet Interval (seconds)
    4. Maximum Packet Interval (seconds)
    5. Minimum Packet Interval (seconds)
    6. Average Packet Length (bytes)
    7. Maximum Packet Length (bytes)
    8. Minimum Packet Length (bytes)
    9. Most Common Packet Length (bytes)
    
    Args:
        packet_df: DataFrame with packet-level data (must have 'second' column)
    
    Returns:
        DataFrame with one row per 1-second window and 9 features + label
    """
    flow_records = []
    
    # Group by second window and label
    for (second, label), group in packet_df.groupby(['second', 'label']):
        # Sort by timestamp within the window
        group = group.sort_values('time')
        
        # Extract packet lengths
        lengths = group['length'].values
        
        # Compute inter-packet intervals (time between consecutive packets)
        intervals = group['time'].diff().dropna().values
        
        # Find most common packet length
        most_common_length = Counter(lengths).most_common(1)[0][0] if len(lengths) else 0
        
        # Compute all 9 features
        flow_records.append({
            'Second_Window':             second,
            'Packet_Count':              len(group),
            'Total_Length':              int(lengths.sum()),
            'Average_Packet_Interval':   round(float(intervals.mean()), 6) if len(intervals) else 0,
            'Maximum_Packet_Interval':   round(float(intervals.max()), 6) if len(intervals) else 0,
            'Minimum_Packet_Interval':   round(float(intervals.min()), 6) if len(intervals) else 0,
            'Average_Packet_Length':     round(float(lengths.mean()), 2),
            'Maximum_Packet_Length':     int(lengths.max()),
            'Minimum_Packet_Length':     int(lengths.min()),
            'Most_Common_Packet_Length': int(most_common_length),
            'Label':                     label,
        })
    
    return pd.DataFrame(flow_records)


# ========== Main Processing ==========
if __name__ == "__main__":
    print("="*60)
    print("Packet Extraction and Feature Computation")
    print("="*60)
    
    # File paths - update these to match your CSV locations
    files = [
        ('amazon_capture.csv', 'amazon'),
        ('hulu_capture.csv', 'hulu'),
        ('youtube_capture.csv', 'youtube'),
    ]
    
    all_flows = []
    
    for csv_path, label in files:
        print(f"\nProcessing {csv_path}...")
        
        # Extract packets from CSV
        packets = extract_packets_from_csv(csv_path, label)
        
        # Compute flow features
        flows = compute_flow_features(packets)
        
        print(f"  Generated {len(flows)} one-second windows")
        all_flows.append(flows)
    
    # Combine all three datasets
    combined = pd.concat(all_flows, ignore_index=True)
    
    print("\n" + "="*60)
    print("COMBINED DATASET SUMMARY")
    print("="*60)
    print(f"Total windows: {len(combined)}")
    print(f"\nLabel distribution:")
    print(combined['Label'].value_counts().to_string())
    
    # Save to CSV
    output_path = 'combined_second_flows.csv'
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    print("\n" + "="*60)
    print("METHODOLOGY NOTE FOR YOUR REPORT")
    print("="*60)
    print("""
This approach is IP-free and time-based:
- No IP addresses are used for grouping or classification
- Each 1-second window is treated as an independent observation
- Label comes purely from which capture file the packets belong to
- This makes the method more practical for real deployment (no NAT/VPN issues)

Key design choice:
- floor(timestamp) assigns packets to discrete 1-second bins
- Overlapping windows would create data leakage between train/test
- Non-overlapping windows ensure true independence of observations
    """)
