import os
import logging
import subprocess
import tempfile

DEFAULT_TIMEOUT = 120

class ScriptExecutor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, huggingface_key: str = None):
        self.timeout = timeout
        self.huggingface_key = huggingface_key or os.environ.get("HUGGINGFACE_API_KEY")

        logging.info(f"ScriptExecutor initialized with huggingface_key and timeout: {self.timeout}s")

    def execute_script(self, script_content: str, working_dir: str = None) -> dict:

        logging.info(f"Attempting to execute script...")

        original_cwd = os.getcwd()
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)

        temp_script = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", dir=os.getcwd()) as sf:
                sf.write(script_content)
                temp_script = sf.name

            env = os.environ.copy()
            if self.huggingface_key:
                env["HUGGINGFACE_API_KEY"] = self.huggingface_key

            result = subprocess.run(["python", temp_script], env=env, capture_output=True, text=True, check=True)

            logging.debug(f"STDOUT:\n{result.stdout}")
            logging.debug(f"STDERR:\n{result.stderr}")

            if result.returncode == 0:
                return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
            else:
                error_msg = f"Script execution failed with return code: {result.returncode}.\nSTDERR:\n{result.stderr}"
                return {"status": "error", **error_msg}

        except Exception as e:
            error_msg = f"Script execution failed with exception: {e}"
            return {"status": "error", **error_msg}

        except subprocess.TimeoutExpired:
            error_msg = f"Script execution timed out after {self.timeout} seconds."
            return {"status": "error", **error_msg}

        finally:
            os.chdir(original_cwd)
            if temp_script and os.path.exists(temp_script):
                os.remove(temp_script)


