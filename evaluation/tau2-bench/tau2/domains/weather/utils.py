from tau2.utils.utils import DATA_DIR

WEATHER_DATA_DIR = DATA_DIR / "tau2" / "domains" / "weather"
WEATHER_DB_PATH = WEATHER_DATA_DIR / "db.json"
WEATHER_POLICY_PATH = WEATHER_DATA_DIR / "policy.md"
# Not all repo variants ship a tasks file for this domain; tasks are typically provided via CLI.
WEATHER_TASK_SET_PATH = WEATHER_DATA_DIR / "tasks.json"
