from tau2.utils.utils import DATA_DIR

RAILWAY_DATA_DIR = DATA_DIR / "tau2" / "domains" / "railway"
RAILWAY_DB_PATH = RAILWAY_DATA_DIR / "db.json"
RAILWAY_POLICY_PATH = RAILWAY_DATA_DIR / "policy.md"
# Not all repo variants ship a tasks file for this domain; tasks are typically provided via CLI.
RAILWAY_TASK_SET_PATH = RAILWAY_DATA_DIR / "tasks.json"
