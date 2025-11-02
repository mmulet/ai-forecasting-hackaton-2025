#!/usr/bin/env python3
"""
Analyze token usage across evaluation logs
"""

from pathlib import Path
from inspect_ai.log import read_eval_log
import sys

def analyze_token_usage(task_name=None):
    """Analyze token usage from logs"""
    logs_dir = Path("logs")
    log_files = sorted(logs_dir.glob("*.eval"), reverse=True)
    
    if task_name:
        log_files = [f for f in log_files if task_name in f.name]
    
    if not log_files:
        print("No log files found")
        return
    
    print("="*80)
    print("TOKEN USAGE ANALYSIS")
    print("="*80)
    print()
    
    for log_file in log_files:
        try:
            log = read_eval_log(log_file)
            
            print(f"\nLog: {log_file.name}")
            print(f"Task: {log.eval.task}, Model: {log.eval.model}")
            print(f"Samples: {len(log.samples)}")
            
            # Overall stats
            if hasattr(log, 'stats') and log.stats:
                stats = log.stats
                if hasattr(stats, 'model_usage') and stats.model_usage:
                    print("\n  Overall Token Usage:")
                    for model_name, usage in stats.model_usage.items():
                        print(f"    Model: {model_name}")
                        if hasattr(usage, 'input_tokens'):
                            print(f"      Input tokens:  {usage.input_tokens:,}")
                        if hasattr(usage, 'output_tokens'):
                            print(f"      Output tokens: {usage.output_tokens:,}")
                        if hasattr(usage, 'total_tokens'):
                            print(f"      Total tokens:  {usage.total_tokens:,}")
            
            # Per-sample analysis
            total_input = 0
            total_output = 0
            samples_with_usage = 0
            
            for sample in log.samples:
                if hasattr(sample, 'output') and sample.output and hasattr(sample.output, 'usage'):
                    usage = sample.output.usage
                    if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                        total_input += usage.input_tokens
                        total_output += usage.output_tokens
                        samples_with_usage += 1
            
            if samples_with_usage > 0:
                print(f"\n  Per-Sample Averages ({samples_with_usage} samples):")
                print(f"    Avg input tokens:  {total_input / samples_with_usage:,.1f}")
                print(f"    Avg output tokens: {total_output / samples_with_usage:,.1f}")
                print(f"    Avg total tokens:  {(total_input + total_output) / samples_with_usage:,.1f}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"{log_file.name}: Error - {e}\n")

if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_token_usage(task_name)
