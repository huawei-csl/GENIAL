from genial.globals import global_vars
from dotenv import load_dotenv
import importlib.util

load_dotenv()

# Detect optional dependency availability without importing it
if importlib.util.find_spec("FLOWY") is not None:
    global_vars["flowy_available"] = True
else:
    global_vars["flowy_available"] = False
