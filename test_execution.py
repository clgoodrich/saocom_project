"""
Execute saocom_analysis_complete.ipynb and report results
Uses papermill for notebook execution with error capture
"""
import subprocess
import sys
from pathlib import Path
import time

def execute_notebook():
    """Execute notebook using jupyter nbconvert"""

    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_complete.ipynb')
    output_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_complete_executed.ipynb')

    if not nb_path.exists():
        print(f"ERROR: Notebook not found: {nb_path}")
        return False

    print("=" * 80)
    print("EXECUTING SAOCOM_ANALYSIS_COMPLETE.IPYNB")
    print("=" * 80)
    print(f"Input: {nb_path.name}")
    print(f"Output: {output_path.name}")
    print()
    print("This will execute all cells and may take several minutes...")
    print("Press Ctrl+C to abort")
    print()

    start_time = time.time()

    try:
        # Execute notebook
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',  # Execute in place
            '--ExecutePreprocessor.timeout=600',  # 10 minute timeout per cell
            str(nb_path)
        ]

        print("Running command:")
        print(" ".join(cmd))
        print()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=nb_path.parent
        )

        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print("EXECUTION RESULTS")
        print("=" * 80)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print()

        if result.returncode == 0:
            print("[OK] NOTEBOOK EXECUTED SUCCESSFULLY")
            print()
            print("All cells executed without errors")
            print()
            return True
        else:
            print("[X] EXECUTION FAILED")
            print()
            print("STDOUT:")
            print(result.stdout)
            print()
            print("STDERR:")
            print(result.stderr)
            print()
            return False

    except KeyboardInterrupt:
        print()
        print("Execution interrupted by user")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = execute_notebook()
    sys.exit(0 if success else 1)
