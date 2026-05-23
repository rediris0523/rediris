import asyncio
import subprocess
import os
import time
import threading
import sys
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class AutoUpdateService:

    def __init__(
        self,
        github_repo: str,
        branch: str = "main",
        check_interval: int = 300,
        restart_delay: int = 10,
        allowed_repo_patterns: Optional[List[str]] = None,
        require_commit_verification: bool = True,
        max_updates_per_hour: int = 2,
        enable_rollback: bool = True,
        script_whitelist: Optional[List[str]] = None
    ):
        self.github_repo = github_repo
        self.branch = branch
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        self.require_commit_verification = require_commit_verification
        self.max_updates_per_hour = max_updates_per_hour
        self.enable_rollback = enable_rollback
        
        self.allowed_repo_patterns = allowed_repo_patterns or [
            "moiraiNewyork/*",
        ]
        
        self.script_whitelist = script_whitelist or [
            "setup_env.sh",
            "run_miner.sh",
            "run_validator.sh",
            "run_task_center.sh"
        ]
        
        self.script_dir = self._detect_script_directory()
        self.neuron_type = self._detect_neuron_type_safe()
        self.project_root = self._detect_project_root()
        
        self.current_commit: Optional[str] = None
        self.is_running = False
        self._check_thread: Optional[threading.Thread] = None
        self._update_in_progress = threading.Lock()
        
        self._update_history: List[datetime] = []
        self._update_history_lock = threading.Lock()
        
        self._backup_commits: List[Tuple[str, datetime]] = []
        self._backup_commits_lock = threading.Lock()
        
        self._validate_repository()
    
    def _validate_repository(self):
        repo_valid = False
        for pattern in self.allowed_repo_patterns:
            if self._match_pattern(self.github_repo, pattern):
                repo_valid = True
                break
        
        if not repo_valid:
            raise ValueError(
                f"Repository '{self.github_repo}' does not match allowed patterns: {self.allowed_repo_patterns}"
            )
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            try:
                remote_url = self._execute_git_command(
                    ["config", "--get", "remote.origin.url"],
                    check=False
                )
                if remote_url:
                    if "github.com" not in remote_url.lower() and not remote_url.startswith("git@"):
                        logger.warning(f"Suspicious remote URL: {remote_url}")
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            logger.warning(f"Could not validate remote URL: {e}")
    
    def _match_pattern(self, text: str, pattern: str) -> bool:
        if '*' not in pattern:
            return text == pattern
        parts = pattern.split('*')
        if len(parts) == 2:
            return text.startswith(parts[0]) and text.endswith(parts[1])
        return False
    
    def _detect_script_directory(self) -> Path:
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, '__file__'):
            script_path = Path(main_module.__file__).resolve()
            if script_path.is_absolute() and script_path.exists():
                return script_path.parent
            else:
                logger.warning(f"Invalid script path detected: {script_path}")
        
        return Path.cwd().resolve()
    
    def _detect_neuron_type_safe(self) -> str:
        script_dir_resolved = self.script_dir.resolve()
        script_dir_str = str(script_dir_resolved)
        
        if 'task_center' in script_dir_str:
            return 'task_center'
        elif 'validator' in script_dir_str:
            return 'validator'
        elif 'miner' in script_dir_str:
            return 'miner'
        
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, '__file__'):
            script_name = Path(main_module.__file__).stem
            if 'task_center' in script_name:
                return 'task_center'
            elif 'validator' in script_name:
                return 'validator'
            elif 'miner' in script_name:
                return 'miner'
        
        if 'task_center' in str(self.script_dir):
            return 'task_center'
        elif 'validator' in str(self.script_dir):
            return 'validator'
        elif 'miner' in str(self.script_dir):
            return 'miner'
        
        logger.warning(f"Could not detect neuron type, defaulting to task_center")
        return 'task_center'
    
    def _detect_project_root(self) -> Path:
        current = self.script_dir.resolve()
        max_depth = 10
        depth = 0
        
        while current != current.parent and depth < max_depth:
            setup_py = current / 'setup.py'
            moirai_dir = current / 'rediris'
            
            if setup_py.exists() and setup_py.is_file():
                if moirai_dir.exists() and moirai_dir.is_dir():
                    return current
            current = current.parent
            depth += 1
        
        return self.script_dir.parent.resolve()
    
    def _check_update_rate_limit(self) -> bool:
        with self._update_history_lock:
            now = datetime.now()
            self._update_history = [
                dt for dt in self._update_history
                if now - dt < timedelta(hours=1)
            ]
            
            if len(self._update_history) >= self.max_updates_per_hour:
                logger.warning(
                    f"Update rate limit exceeded: {len(self._update_history)} updates in the last hour "
                    f"(max: {self.max_updates_per_hour})"
                )
                return False
            
            return True
    
    def _record_update(self):
        with self._update_history_lock:
            self._update_history.append(datetime.now())
    
    async def start(self):
        if self.is_running:
            logger.warning("Auto-update service is already running")
            return
        
        self.is_running = True
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="auto-updater"
        )
        self._check_thread.start()
        logger.info(
            f"Auto-update service started (checking every {self.check_interval}s, "
            f"neuron_type={self.neuron_type}, rate_limit={self.max_updates_per_hour}/hour)"
        )
    
    async def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=5.0)
        logger.info("Auto-update service stopped")
    
    def _monitor_loop(self):
        while self.is_running:
            try:
                self._perform_update_check()
            except Exception as e:
                logger.error(f"Error in update check loop: {e}", exc_info=True)
            
            sleep_count = 0
            while self.is_running and sleep_count < self.check_interval:
                time.sleep(1)
                sleep_count += 1
    
    def _perform_update_check(self):
        if not self._update_in_progress.acquire(blocking=False):
            logger.debug("Update check skipped - update already in progress")
            return
        
        try:
            if not self._check_update_rate_limit():
                return
            
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            
            try:
                current_branch = self._execute_git_command(
                    ["rev-parse", "--abbrev-ref", "HEAD"]
                ).strip()
                
                if current_branch != self.branch:
                    logger.warning(
                        f"Current branch '{current_branch}' does not match configured branch '{self.branch}'"
                    )
                    return
                
                local_commit = self._execute_git_command(["rev-parse", "HEAD"]).strip()
                
                logger.debug("Fetching latest changes from remote...")
                self._execute_git_command(["fetch", "origin"], check=False)
                
                remote_ref = f"origin/{current_branch}"
                remote_commit = self._execute_git_command(["rev-parse", remote_ref]).strip()
                
                if not self._is_valid_commit_hash(remote_commit):
                    logger.error(f"Invalid remote commit hash: {remote_commit}")
                    return
                
                if self._needs_update(local_commit, remote_commit):
                    logger.info(
                        f"Update detected: local={local_commit[:8]}, remote={remote_commit[:8]}"
                    )
                    self._apply_update(remote_commit, current_branch)
                else:
                    if self.current_commit is None:
                        self.current_commit = local_commit
                        logger.info(f"Initial commit: {local_commit[:8]}")
                    else:
                        logger.debug(f"Repository is up-to-date (commit: {local_commit[:8]})")
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Update check failed: {e}", exc_info=True)
        finally:
            self._update_in_progress.release()
    
    def _is_valid_commit_hash(self, commit_hash: str) -> bool:
        return len(commit_hash) == 40 and all(c in '0123456789abcdef' for c in commit_hash.lower())
    
    def _needs_update(self, local_commit: str, remote_commit: str) -> bool:
        return local_commit != remote_commit
    
    def _verify_commit(self, commit_hash: str) -> bool:
        if not self.require_commit_verification:
            return True
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            try:
                verify_result = self._execute_git_command(
                    ["verify-commit", commit_hash],
                    check=False
                )
                
                if verify_result is None:
                    logger.warning(f"Commit {commit_hash[:8]} verification failed or not signed")
                    return False
                
                logger.debug(f"Commit {commit_hash[:8]} verified successfully")
                return True
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            logger.warning(f"Could not verify commit signature: {e}")
            return not self.require_commit_verification
    
    def _create_backup(self, commit_hash: str) -> bool:
        if not self.enable_rollback:
            return True
        
        try:
            with self._backup_commits_lock:
                if len(self._backup_commits) >= 5:
                    self._backup_commits.pop(0)
                
                self._backup_commits.append((commit_hash, datetime.now()))
                logger.info(f"Backup created for commit {commit_hash[:8]}")
                return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _rollback_update(self) -> bool:
        if not self.enable_rollback:
            return False
        
        try:
            with self._backup_commits_lock:
                if not self._backup_commits:
                    logger.error("No backup commits available for rollback")
                    return False
                
                backup_commit, backup_time = self._backup_commits[-1]
                
                logger.warning(f"Attempting rollback to commit {backup_commit[:8]} (backed up at {backup_time})")
                
                original_cwd = os.getcwd()
                os.chdir(self.script_dir)
                try:
                    self._execute_git_command(
                        ["reset", "--hard", backup_commit],
                        capture_output=False,
                        check=True
                    )
                    logger.info(f"Successfully rolled back to commit {backup_commit[:8]}")
                    self.current_commit = backup_commit
                    return True
                finally:
                    os.chdir(original_cwd)
        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return False
    
    def _apply_update(self, target_commit: str, branch: str):
        logger.info(f"Applying update to commit {target_commit[:8]}...")
        
        if not self._verify_commit(target_commit):
            logger.error(f"Commit verification failed for {target_commit[:8]}, aborting update")
            return
        
        current_commit = self._execute_git_command(["rev-parse", "HEAD"]).strip()
        if not self._create_backup(current_commit):
            logger.warning("Failed to create backup, but continuing with update")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            
            try:
                reset_cmd = ["git", "reset", "--hard", target_commit]
                self._execute_git_command(reset_cmd, capture_output=False, check=True)
                logger.info(f"Successfully updated to commit {target_commit[:8]}")
                
                self.current_commit = target_commit
                
                try:
                    self._run_post_update_steps()
                except Exception as e:
                    logger.error(f"Post-update steps failed: {e}")
                    if self.enable_rollback:
                        logger.info("Attempting rollback due to post-update failure...")
                        self._rollback_update()
                    raise
                
                self._record_update()
                
                logger.info(f"Scheduling restart in {self.restart_delay} seconds...")
                time.sleep(self.restart_delay)
                
                self._restart_service()
                
            except Exception as e:
                logger.error(f"Failed to apply update: {e}", exc_info=True)
                if self.enable_rollback:
                    logger.info("Attempting rollback due to update failure...")
                    self._rollback_update()
                raise
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Failed to apply update: {e}", exc_info=True)
            raise
    
    def _validate_script_path(self, script_path: Path) -> bool:

        script_name = script_path.name
        if script_name not in self.script_whitelist:
            logger.error(f"Script '{script_name}' is not in whitelist: {self.script_whitelist}")
            return False
        
        if not script_path.exists() or not script_path.is_file():
            logger.error(f"Script path is invalid: {script_path}")
            return False
        
        try:
            script_path.resolve().relative_to(self.project_root.resolve())
        except ValueError:
            logger.error(f"Script path is outside project root: {script_path}")
            return False
        

        if script_path.is_symlink():
            logger.error(f"Script is a symlink, which is not allowed: {script_path}")
            return False

        stat_info = script_path.stat()
        if stat_info.st_mode & 0o002:
            logger.warning(f"Script is world-writable: {script_path}")
        
        return True
    
    def _run_post_update_steps(self):
        logger.info("Running post-update steps...")
        
        try:
            setup_script = self.project_root / "scripts" / "setup_env.sh"
            
            if setup_script.exists():
                if not self._validate_script_path(setup_script):
                    raise ValueError(f"Invalid setup script: {setup_script}")
                
                venv_path = self._find_venv_path()
                
                if venv_path:

                    activate_script = venv_path / "bin" / "activate"
                    
                    if not activate_script.exists() or not activate_script.is_file():
                        raise ValueError(f"Invalid activate script: {activate_script}")
                    
                    logger.info(f"Executing setup script: {setup_script}")
                    
                    cmd = [
                        '/bin/bash',
                        '-c',
                        f'source "{activate_script}" && "{setup_script}"'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(self.project_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=os.environ.copy()
                    )
                    
                    stdout, stderr = process.communicate(timeout=300)
                    
                    if process.returncode == 0:
                        logger.info("Post-update setup completed successfully")
                    else:
                        error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                        logger.warning(
                            f"Setup script returned non-zero exit code: {process.returncode}, "
                            f"error: {error_msg[:200]}"
                        )
                        raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
                else:
                    logger.warning("No virtual environment found, skipping setup script")
            else:
                logger.debug("No setup script found, skipping post-update steps")
                
        except subprocess.TimeoutExpired:
            logger.error("Setup script execution timed out")
            raise
        except Exception as e:
            logger.warning(f"Post-update steps failed: {e}")
            raise
    
    def _find_venv_path(self) -> Optional[Path]:
        venv_paths = [
            self.project_root.parent / "venv_moirai",
            self.project_root / "venv",
            self.project_root.parent / "venv"
        ]
        
        for path in venv_paths:
            path_resolved = path.resolve()
            activate_script = path_resolved / "bin" / "activate"
            
            if path_resolved.exists() and path_resolved.is_dir():
                if activate_script.exists() and activate_script.is_file():
                    if not activate_script.is_symlink():
                        return path_resolved
        
        return None
    
    def _restart_service(self):
        logger.info("Initiating service restart...")
        
        try:
            start_script = self.project_root / "scripts" / f"run_{self.neuron_type}.sh"
            
            if start_script.exists():
                if not self._validate_script_path(start_script):
                    logger.warning("Start script validation failed, using direct Python restart")
                    self._restart_via_python()
                    return
                
                venv_path = self._find_venv_path()
                
                if venv_path:
                    activate_script = venv_path / "bin" / "activate"
                    
                    cmd = [
                        '/bin/bash',
                        '-c',
                        f'source "{activate_script}" && "{start_script}"'
                    ]
                    
                    logger.info(f"Executing start script: {start_script}")
                    subprocess.Popen(
                        cmd,
                        cwd=str(self.project_root),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                        env=os.environ.copy()
                    )
                    
                    time.sleep(2)
                    
                    logger.info("Exiting current process...")
                    sys.exit(0)
                else:
                    logger.warning("No virtual environment found, using direct Python restart")
                    self._restart_via_python()
            else:
                logger.info("No start script found, using direct Python restart")
                self._restart_via_python()
                
        except Exception as e:
            logger.error(f"Failed to restart via script: {e}, attempting direct Python restart")
            self._restart_via_python()
    
    def _restart_via_python(self):
        try:
            logger.info("Restarting via Python process re-execution...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart Python process: {e}")
            raise
    
    def _execute_git_command(
        self,
        args: List[str],
        capture_output: bool = True,
        check: bool = True
    ) -> Optional[str]:

        if not args or not isinstance(args, list):
            raise ValueError("Invalid git command arguments")
        
        dangerous_commands = ['config', 'remote', 'filter-branch', 'update-ref']
        if any(arg in dangerous_commands for arg in args):
            if not (args[0] == 'config' and '--get' in args):
                logger.warning(f"Potentially dangerous git command blocked: {args}")
                raise ValueError(f"Dangerous git command not allowed: {args[0]}")
        
        cmd = ["git"] + args
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=check,
                    cwd=str(self.script_dir)
                )
                if check or result.returncode == 0:
                    return result.stdout.strip() if result.stdout else ""
                return None
            else:
                result = subprocess.run(
                    cmd,
                    timeout=60,
                    check=check,
                    cwd=str(self.script_dir)
                )
                return None
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}, error: {e}")
            if check:
                raise
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing git command: {e}")
            raise
