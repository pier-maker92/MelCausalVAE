import os
import sys
import yaml
import subprocess
import time
import re
import argparse
from queue import Queue
from threading import Thread
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def construct_command(job, global_cfg, is_vocoder=False):
    # Same as before
    if is_vocoder:
        script = "eval_vocoder.py"
        model_arg = ["--vocoder-name", job['vocoder_name']]
        exp_name = f"{job['vocoder_name']}_{global_cfg['dataset_setting'].lower()}_eval"
    else:
        script = "eval.py"
        model_arg = ["--baseline-model", job['model_name']]
        
        hz_val = "native"
        for i, a in enumerate(job.get('extra_args', [])):
            if a == "--baseline-hz" and i + 1 < len(job.get('extra_args', [])):
                hz_val = job.get('extra_args', [])[i+1]
        
        exp_name = f"{job['model_name']}_{hz_val}hz_{global_cfg['dataset_setting'].lower()}_eval"
    
    cmd = ["python", "-u", script] + model_arg # Force unbuffered output for real-time parsing
    
    if global_cfg.get('dataset_setting'):
        cmd.extend(["--setting", global_cfg['dataset_setting']])
    if global_cfg.get('filter_librispeech'):
        cmd.append("--filter-librispeech")
    if 'num_samples' in global_cfg:
        cmd.extend(["--num-samples", str(global_cfg['num_samples'])])
    if 'batch_size' in global_cfg:
        cmd.extend(["--batch-size", str(global_cfg['batch_size'])])
    if global_cfg.get('utmos'):
        cmd.append("--UTMOS")
    if global_cfg.get('output_dir'):
        cmd.extend(["--output-dir", global_cfg['output_dir']])
        
    cmd.extend(["--exp-name", exp_name])
    
    if 'extra_args' in job:
        cmd.extend(job['extra_args'])
        
    return cmd

def worker_thread(gpu_id, worker_idx, job_queue, resources_cfg, progress, task_ids):
    threads_per_job = str(resources_cfg.get('threads_per_job', 4))
    cores_per_job = resources_cfg.get('cores_per_job', 4)
    start_core = worker_idx * cores_per_job
    end_core = start_core + cores_per_job - 1
    cpu_mask = f"{start_core}-{end_core}"

    task_id = task_ids[worker_idx]

    while not job_queue.empty():
        job_info = job_queue.get()
        cmd = job_info['cmd']
        name = job_info['name']
        
        progress.update(task_id, description=f"[cyan]GPU {gpu_id}[/cyan]: {name}", completed=0, total=100)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = threads_per_job
        env["MKL_NUM_THREADS"] = threads_per_job
        env["OPENBLAS_NUM_THREADS"] = threads_per_job
        env["NUMEXPR_NUM_THREADS"] = threads_per_job
        env["VECLIB_MAXIMUM_THREADS"] = threads_per_job
        
        full_cmd = ["taskset", "-c", cpu_mask] + cmd
        
        log_file_path = f"run_{name}_gpu{gpu_id}.log"
        
        try:
            # We use Popen with subprocess.PIPE to read output line-by-line in real-time
            process = subprocess.Popen(
                full_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Simple regex to catch tqdm progress lines like "Evaluating dataset:  23%|...| 256/1132"
            progress_pattern = re.compile(r'(\d+)%\|')
            
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Command: {' '.join(full_cmd)}\n")
                log_file.write(f"CUDA_VISIBLE_DEVICES: {gpu_id}\n")
                log_file.write(f"CPUs: {cpu_mask}\n\n")
                
                # In Python, when reading from a PIPE that gets carriage returns \r without \n (typical of tqdm),
                # readline() might block or read character by character incorrectly. Iterating over stdout works better.
                while True:
                    # Read character by character since tqdm uses \r without \n
                    char = process.stdout.read(1)
                    if not char and process.poll() is not None:
                        break
                    
                    if char:
                        log_file.write(char)
                        log_file.flush()
                        
                        # Buffer a line manually until \r or \n
                        if not hasattr(process.stdout, 'line_buffer'):
                            process.stdout.line_buffer = ""
                            
                        if char == '\r' or char == '\n':
                            match = progress_pattern.search(process.stdout.line_buffer)
                            if match:
                                pct = int(match.group(1))
                                progress.update(task_id, completed=pct)
                            process.stdout.line_buffer = ""
                        else:
                            process.stdout.line_buffer += char

            process.wait()
            
            if process.returncode == 0:
                progress.update(task_id, description=f"[green]GPU {gpu_id}[/green]: {name} (Done)", completed=100)
            else:
                progress.update(task_id, description=f"[red]GPU {gpu_id}[/red]: {name} (Error!)")
                
        except Exception as e:
            progress.update(task_id, description=f"[red]GPU {gpu_id}[/red]: {name} (Exception!)")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\nException: {str(e)}\n")
                
        job_queue.task_done()
        time.sleep(2) # brief pause before starting next job on this worker

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Evaluation Orchestrator")
    parser.add_argument("config", nargs='?', default=None, help="Path to config YAML file")
    parser.add_argument("--config", dest="config_flag", help="Path to config YAML file")
    args = parser.parse_args()

    config_file = args.config_flag if args.config_flag else args.config
    if not config_file:
        parser.print_help()
        sys.exit(1)

    cfg = load_config(config_file)
    
    global_cfg = cfg.get('global_settings', {})
    resources_cfg = cfg.get('resources', {})
    gpu_ids = resources_cfg.get('gpu_ids', [0])
    
    jobs = cfg.get('jobs', {})
    baselines = jobs.get('baselines', [])
    vocoders = jobs.get('vocoders', [])
    
    job_queue = Queue()
    
    for b in baselines:
        cmd = construct_command(b, global_cfg, is_vocoder=False)
        job_queue.put({'name': f"Baseline_{b['model_name']}", 'cmd': cmd})
        
    for v in vocoders:
        cmd = construct_command(v, global_cfg, is_vocoder=True)
        job_queue.put({'name': f"Vocoder_{v['vocoder_name']}", 'cmd': cmd})
        
    print(f"Loaded {job_queue.qsize()} jobs. Starting {len(gpu_ids)} workers.")
    
    # Initialize rich Progress
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    
    task_ids = []
    for gpu_id in gpu_ids:
        # Create a progress bar task for each worker
        task_id = progress.add_task(f"[cyan]GPU {gpu_id}[/cyan]: Waiting...", total=100)
        task_ids.append(task_id)

    threads = []
    
    with Live(progress, refresh_per_second=4) as live:
        for worker_idx, gpu_id in enumerate(gpu_ids):
            t = Thread(target=worker_thread, args=(gpu_id, worker_idx, job_queue, resources_cfg, progress, task_ids))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
        
    print("-" * 50)
    print("All evaluation jobs completed.")

if __name__ == "__main__":
    main()
