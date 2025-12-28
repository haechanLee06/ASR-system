import sys
import logging
import traceback
import importlib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("diagnose_diarization")

def check_dependencies():
    logger.info("Checking dependencies...")
    print(f"Python executable: {sys.executable}")
    
    dependencies = [
        ("modelscope", "modelscope"),
        ("funasr", "funasr"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("cryptography", "cryptography"),
        ("pyOpenSSL", "OpenSSL"),
        ("urllib3", "urllib3")
    ]
    
    for pkg_name, module_name in dependencies:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "Unknown")
            print(f"{pkg_name}: {version}")
        except ImportError:
            print(f"{pkg_name}: Not Installed")
        except Exception as e:
            print(f"{pkg_name}: Error checking version - {e}")

def test_pipeline():
    logger.info("Starting isolated loading test for 3D-Speaker Pipeline...")
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        logger.info("Imported modelscope successfully. Attempting to load pipeline...")
        
        p = pipeline(
            task=Tasks.speaker_diarization,
            model='damo/speech_campplus_sv_zh_en_16k-common_advanced',
            model_revision='v1.0.0'
        )
        logger.info("Pipeline loaded successfully!")
        print("SUCCESS: 3D-Speaker pipeline initialized without errors.")
        
    except Exception:
        logger.error("Failed to load pipeline.")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Diagnostic Tool Start ===")
    check_dependencies()
    print("-" * 30)
    test_pipeline()
    print("=== Diagnostic Tool End ===")
