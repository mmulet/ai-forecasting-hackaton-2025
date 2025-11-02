#!/usr/bin/env python3
"""Read and analyze Inspect AI evaluation logs"""

import json
from pathlib import Path
from inspect_ai.log import read_eval_log

def read_latest_log(task_name=None, index=0):
    """Read the most recent log file (or specify index for older logs)"""
    logs_dir = Path("logs")
    
    # Get all .eval files
    log_files = sorted(logs_dir.glob("*.eval"), reverse=True)
    
    if task_name:
        # Filter by task name
        log_files = [f for f in log_files if task_name in f.name]
    
    if not log_files:
        print("No log files found")
        return
    
    # Show available logs if multiple
    if len(log_files) > 1:
        print(f"Found {len(log_files)} matching logs (showing #{index}):")
        for i, f in enumerate(log_files[:5]):  # Show first 5
            marker = " <-- READING THIS ONE" if i == index else ""
            print(f"  [{i}] {f.name}{marker}")
        print()
    
    if index >= len(log_files):
        print(f"Index {index} out of range (only {len(log_files)} logs)")
        return
    
    latest = log_files[index]
    print(f"Reading: {latest.name}\n")
    
    # Read the log
    log = read_eval_log(latest)
    
    # Print summary
    print(f"Task: {log.eval.task}")
    print(f"Model: {log.eval.model}")
    print(f"Status: {log.status}")
    print(f"Samples: {len(log.samples)}")
    
    # Print token usage stats
    if hasattr(log, 'stats') and log.stats:
        print(f"\nToken Usage Statistics:")
        stats = log.stats
        if hasattr(stats, 'model_usage') and stats.model_usage:
            for model_name, usage in stats.model_usage.items():
                print(f"  Model: {model_name}")
                if hasattr(usage, 'input_tokens'):
                    print(f"    Input tokens: {usage.input_tokens:,}")
                if hasattr(usage, 'output_tokens'):
                    print(f"    Output tokens: {usage.output_tokens:,}")
                if hasattr(usage, 'total_tokens'):
                    print(f"    Total tokens: {usage.total_tokens:,}")
    
    if log.results:
        print(f"\nResults:")
        for metric_name, metric_value in log.results.metrics.items():
            print(f"  {metric_name}: {metric_value.value}")
    
    # Print sample details
    print(f"\nSample Details:")
    for i, sample in enumerate(log.samples[:5]):  # First 5 samples only
        print(f"\n{'='*80}")
        print(f"Sample {i+1}:")
        print(f"{'='*80}")
        
        # Show the full message history/conversation
        if hasattr(sample, 'messages') and sample.messages:
            print(f"\n--- MESSAGES/PROMPTS ---")
            print(f"Total messages: {len(sample.messages)}")
            
            # Count assistant messages (model responses)
            assistant_count = sum(1 for msg in sample.messages if hasattr(msg, 'role') and msg.role == 'assistant')
            print(f"Assistant messages (model turns): {assistant_count}")
            
            for j, msg in enumerate(sample.messages):
                role = msg.role if hasattr(msg, 'role') else 'unknown'
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"\n[Message {j+1} - {role.upper()}]:")
                print(content)
        
        print(f"\n--- INPUT ---")
        print(sample.input)  # Full input, no truncation
        
        if sample.output:
            print(f"\n--- OUTPUT/COMPLETION ---")
            print(sample.output.completion)  # Full output, no truncation
        
        print(f"\n--- SCORE ---")
        print(sample.scores)
        
        # Show token usage for this sample
        if hasattr(sample, 'output') and sample.output and hasattr(sample.output, 'usage'):
            usage = sample.output.usage
            print(f"\n--- TOKEN USAGE (This Sample) ---")
            if hasattr(usage, 'input_tokens'):
                print(f"  Input tokens: {usage.input_tokens:,}")
            if hasattr(usage, 'output_tokens'):
                print(f"  Output tokens: {usage.output_tokens:,}")
            if hasattr(usage, 'total_tokens'):
                print(f"  Total tokens: {usage.total_tokens:,}")
        
        # Show metadata if available
        if hasattr(sample, 'metadata') and sample.metadata:
            print(f"\n--- METADATA ---")
            print(sample.metadata)
    
    return log

def list_all_logs():
    """List all log files with basic info"""
    logs_dir = Path("logs")
    log_files = sorted(logs_dir.glob("*.eval"), reverse=True)
    
    print("Available logs:\n")
    for log_file in log_files:
        try:
            log = read_eval_log(log_file)
            status = log.status
            samples = len(log.samples)
            print(f"{log_file.name}")
            print(f"  Task: {log.eval.task}, Model: {log.eval.model}")
            print(f"  Status: {status}, Samples: {samples}")
            if log.results:
                metrics = log.results.metrics
                for name, val in metrics.items():
                    print(f"  {name}: {val.value}")
            print()
        except Exception as e:
            print(f"{log_file.name}: Error reading - {e}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_all_logs()
        else:
            # Read specific task name, optionally with index
            task_name = sys.argv[1]
            index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            read_latest_log(task_name, index)
    else:
        # Read latest log
        read_latest_log()
