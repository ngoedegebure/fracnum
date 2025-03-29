from dotenv import load_dotenv
import os
load_dotenv('environment.env')  # Load variables from .env
USE_CUPY = os.getenv("USE_CUPY", "0") == "1"

if USE_CUPY:
    import cupy as np
else:
    import numpy as np