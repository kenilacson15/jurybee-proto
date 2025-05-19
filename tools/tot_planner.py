import sys
import os
import json
import logging
import time
from typing import List, Dict, Any

# Project root adjustment
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except Exception as e:
    raise RuntimeError(f"Failed to set PROJECT_ROOT: {e}")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [ToT] %(message)s',
    handlers=[
        logging.FileHandler("tot_planner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom error
class ToTPlannerError(Exception):
    """Base exception for ToTPlanner failures."""

# Import agents and models with error checks
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as e:
    logger.critical("Missing dependency sentence-transformers: %s", e)
    raise

try:
    from langchain_core.exceptions import OutputParserException
    from agents.risk_analyst import RiskAnalystAgent, RiskAnalysisError
    from agents.compliance_checker import ComplianceCheckerAgent, ComplianceError, ComplianceResult
except ImportError as e:
    logger.critical("Agent imports failed: %s", e)
    raise

class ToTPlanner:
    def __init__(self, risk_analyst_agent: RiskAnalystAgent):
        """Initialize ToT Planner with agents and evaluation tools."""
        self.risk_analyst = risk_analyst_agent
        self.compliance_checker = ComplianceCheckerAgent()

        # Load embedding model with retry
        for attempt in range(3):
            try:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                break
            except Exception as e:
                logger.error("Attempt %d: Failed to load embedding model: %s", attempt + 1, e, exc_info=True)
                time.sleep(2)
        else:
            raise ToTPlannerError("Embedding model initialization failed after retries")

        self.weights = {'risk': 0.4, 'compliance': 0.6, 'similarity': 0.2}
        self.knowledge_base = self._load_clause_patterns()

    def _load_clause_patterns(self) -> Dict[str, List[str]]:
        """Load legal clause patterns with fallback to defaults."""
        default_patterns = {
            "confidentiality": ["duration", "exceptions", "scope"],
            "non-compete": ["duration", "geographic_scope", "industry"]
        }
        patterns_path = os.path.join(PROJECT_ROOT, "data", "nda_corpus", "patterns.json")

        if not os.path.isfile(patterns_path):
            logger.warning("Patterns file not found at %s. Using defaults.", patterns_path)
            return default_patterns

        try:
            with open(patterns_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Patterns must be a dict of lists.")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid patterns file: %s", e, exc_info=True)
            return default_patterns
        except Exception as e:
            logger.error("Unexpected error loading patterns: %s", e, exc_info=True)
            return default_patterns

    def generate_paths(self, clause: str, n_paths: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple interpretations of a clause."""
        if not clause or not isinstance(clause, str):
            raise ToTPlannerError("Clause must be a non-empty string.")
        if n_paths < 1:
            raise ToTPlannerError("n_paths must be >= 1.")

        logger.info("Generating %d paths for clause: %s...", n_paths, clause[:30])
        paths = [{"clause": clause, "interpretation": "Literal", "transformations": []}]

        try:
            mods = self._generate_legal_modifications(clause, n_paths - 1)
        except Exception as e:
            logger.warning("Modification generation error: %s", e, exc_info=True)
            mods = []

        for mod in mods:
            text = mod.get("text") or clause
            reason = mod.get("reason", "Modified")
            paths.append({"clause": text, "interpretation": reason, "transformations": [mod]})

        return paths

    def _generate_legal_modifications(self, clause: str, limit: int) -> List[Dict[str, str]]:
        """Apply simple legal modifications to clause."""
        mods = []
        lc = clause.lower()
        if "indefinitely" in lc:
            mods.append({"text": clause.replace("indefinitely", "for 5 years"), "reason": "Standard duration"})
            mods.append({"text": clause + " except for public knowledge", "reason": "Public exception"})
        if "disclosure" in lc and "third parties" not in lc:
            mods.append({"text": clause.replace("disclosure", "disclosure to affiliates only"), "reason": "Restricted"})
            mods.append({"text": clause + " with prior consent", "reason": "Consent added"})
        return mods[:limit]

    def evaluate_paths(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate each path and score them."""
        if not paths:
            raise ToTPlannerError("No paths provided for evaluation.")

        results = []
        for idx, path in enumerate(paths, 1):
            text = path.get("clause", "")
            logger.info("Evaluating path %d: %s", idx, path.get("interpretation"))
            
            # Risk analysis
            try:
                risk = self.risk_analyst.analyze_clause(text)
            except RiskAnalysisError as e:
                logger.error("Risk analysis error: %s", e, exc_info=True)
                risk = {"risk_level": "Unknown", "details": str(e)}

            # Compliance check
            try:
                comp = self.compliance_checker.check_compliance(text)
                comp_data = comp.model_dump()
            except ComplianceError as e:
                logger.error("Compliance check error: %s", e, exc_info=True)
                comp_data = {"compliance_status": "Error", "issues": [str(e)]}

            # Similarity
            try:
                sim_score = self._calculate_similarity(text)
            except Exception as e:
                logger.warning("Similarity error: %s", e, exc_info=True)
                sim_score = 0.5

            # Overall score
            score = self._calculate_score(risk, comp_data, sim_score)

            results.append({
                "path": path,
                "risk": risk,
                "compliance": comp_data,
                "similarity": sim_score,
                "score": score
            })

        return sorted(results, key=lambda r: r["score"], reverse=True)

    def _calculate_score(self, risk: Dict[str, Any], compliance: Dict[str, Any], similarity: float) -> float:
        """Compute weighted score."""
        try:
            risk_map = {"Critical": 0.1, "High": 0.3, "Medium": 0.6, "Low": 0.8, "Unknown": 0.5}
            comp_map = {"Compliant": 1.0, "Conditional": 0.7, "Unknown": 0.5, "Non-Compliant": 0.2, "Error": 0.1}
            r = risk_map.get(risk.get("risk_level"), 0.5)
            c = comp_map.get(compliance.get("compliance_status"), 0.5)
            w = self.weights
            return r*w['risk'] + c*w['compliance'] + similarity*w['similarity']
        except Exception as e:
            logger.error("Score calculation failed: %s", e, exc_info=True)
            return 0.0

    def _calculate_similarity(self, clause: str) -> float:
        """Calculate clause similarity against known patterns."""
        patterns = [p for pats in self.knowledge_base.values() for p in pats]
        texts = patterns + [clause]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        try:
            sims = util.cos_sim(embeddings[-1], embeddings[:-1])[0]
            return float(sims.max().item())
        except Exception:
            raise

    def select_best_path(self, clause: str, n_paths: int = 3) -> Dict[str, Any]:
        """Run full planning pipeline."""
        try:
            paths = self.generate_paths(clause, n_paths)
            evaluated = self.evaluate_paths(paths)
            best = evaluated[0]
            logger.info("Best path: %s (Score: %.2f)", best['path']['interpretation'], best['score'])
            return {
                "best_path": best["path"],
                "score": best["score"],
                "risk": best.get("risk"),
                "compliance": best.get("compliance"),
                "alternatives": [p["path"] for p in evaluated[1:]]
            }
        except ToTPlannerError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in select_best_path: %s", e)
            raise ToTPlannerError("Failed to select best path") from e


if __name__ == "__main__":
    from agents.risk_analyst import RiskAnalystAgent
    try:
        agent = RiskAnalystAgent()
        planner = ToTPlanner(agent)
        result = planner.select_best_path("Confidentiality lasts indefinitely for all information disclosed.")
        print(json.dumps(result, indent=2))
    except ToTPlannerError as e:
        logger.critical("Planner failure: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.critical("Runtime error: %s", e)
        sys.exit(1)
