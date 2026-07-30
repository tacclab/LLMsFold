"""Project-wide defaults and fixed thresholds."""

DEFAULT_PDB_FILE = "data/acvr1_R206H_clean.pdb"
DEFAULT_FEW_SHOT_CSV = "data/few_shot_smiles_patent.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_SAMPLES = 5

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_LLM_TEMPERATURE = 0.8

DEFAULT_BOLTZ_REQUEST_TIMEOUT_SECONDS = 400.0
DEFAULT_BOLTZ_MAX_WAIT_SECONDS = 600
DEFAULT_BOLTZ_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_BOLTZ_POLL_HEADER_SECONDS = 300
DEFAULT_BOLTZ_RETRY_ATTEMPTS = 4
DEFAULT_BOLTZ_RETRY_MIN_WAIT_SECONDS = 2.0
DEFAULT_BOLTZ_RETRY_MAX_WAIT_SECONDS = 30.0
DEFAULT_BOLTZ_RECYCLING_STEPS = 3
DEFAULT_BOLTZ_SAMPLING_STEPS = 200
DEFAULT_BOLTZ_DIFFUSION_SAMPLES = 4
DEFAULT_BOLTZ_STEP_SCALE = 1.2
DEFAULT_BOLTZ_WITHOUT_POTENTIALS = True

DEFAULT_PUBCHEM_TIMEOUT_SECONDS = 10.0
DEFAULT_PUBCHEM_LISTKEY_ATTEMPTS = 3
DEFAULT_PUBCHEM_POLL_INTERVAL_SECONDS = 2.0

DEFAULT_P2RANK_OUTPUT_DIRNAME = "p2rank_output"
DEFAULT_DEEPCHEM_POCKET_PAD = 5.0
DEFAULT_POCKET_CONTACT_DISTANCE = 8.0
# A pocket smaller than this on any axis (raw hull dimensions, before the
# margin below is applied) cannot fit a drug-like ligand plus clearance, so
# it is excluded from selection unless no candidate qualifies.
DEFAULT_POCKET_MIN_BOX_ANGSTROM = 8.0
# Isotropic margin added to the *selected* pocket's box only, to account for
# side-chain flexibility during docking.
DEFAULT_POCKET_BOX_MARGIN_ANGSTROM = 5.0
# Docking boxes are expanded by the margin above, but capped here so an
# oversized/noisy pocket detection can't produce an unbounded search box.
DEFAULT_POCKET_MAX_BOX_ANGSTROM = 30.0

DEFAULT_IPTM_THRESHOLD = 0.5
DEFAULT_PLDDT_THRESHOLD = 0.5

SEED_SMILES_LIMIT = 5
CONTEXT_LEADS_WINDOW = 5
NEGATIVE_LEADS_WINDOW = 3
ALREADY_PROPOSED_WINDOW = 10
MORGAN_FINGERPRINT_RADIUS = 2
ADJ_AFFINITY_THRESHOLD = 0.6
SAS_SCORE_MAX = 10.0
# Ertl's synthetic accessibility score is defined on a 1 (trivial) to 10 (very
# hard) scale; 1 anchors the "easiest possible" end of the reward normalization.
SAS_SCORE_MIN = 1.0

UNIFIED_REPORT_FILENAME = "unified_report.csv"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%H:%M:%S"

# Log files persist across runs, so they carry the full date unlike the
# console handler which only shows time-of-day.
DEFAULT_LOG_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR_NAME = "logs"
