#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEDCON-like evaluator with QuickUMLS - Refactored Version

Usage:
    from medcon_evaluator import MedconScore
    
    scorer = MedconScore()
    scores = scorer(refs=references, hyps=predictions)
"""

import os
import json
import time
import subprocess
from typing import Set, List, Optional

import numpy as np


class MedconScore:
    def __init__(self, 
                 umls_installed_dir: str = "<ENTER UMLS PATH>",
                 quickumls_index: str = "<ENTER QUICKUMLS PATH>",
                 threshold: float = 0.5,
                 window: int = 5,
                 semtypes: Optional[List[str]] = None,
                 build_index_if_missing: bool = True,
                 matching: bool = False):
        """
        Initialize MEDCON scorer with QuickUMLS.
        
        Args:
            umls_installed_dir: Path to MetamorphoSys-installed UMLS directory (contains MRCONSO.RRF, MRSTY.RRF)
            quickumls_index: Path to QuickUMLS index directory
            threshold: QuickUMLS similarity threshold (recommend 0.9~0.95)
            window: QuickUMLS window size
            semtypes: Optional UMLS semantic types to keep (e.g., ['T047', 'T121', 'T109'])
            build_index_if_missing: Whether to build QuickUMLS index if missing
            matching: Whether to print matching debug information (default: False)
        """
        self.umls_installed_dir = umls_installed_dir
        self.quickumls_index = quickumls_index
        self.threshold = threshold
        self.window = window
        self.semtypes = semtypes
        self.build_index_if_missing = build_index_if_missing
        self.matching = matching
        
        # Validate and setup
        self._validate_setup()
        
        # Initialize extractor
        self.extractor = self._create_extractor()
    
    def _validate_setup(self):
        """Validate UMLS directory and ensure QuickUMLS index exists."""
        self._assert_umls_rrf_dir(self.umls_installed_dir)
        self._ensure_quickumls_index(
            self.umls_installed_dir, 
            self.quickumls_index, 
            self.build_index_if_missing
        )
    
    def _assert_umls_rrf_dir(self, d: str):
        """Check if UMLS directory contains required RRF files."""
        mc = os.path.join(d, "MRCONSO.RRF")
        ms = os.path.join(d, "MRSTY.RRF")
        if not (os.path.isfile(mc) and os.path.isfile(ms)):
            raise FileNotFoundError(
                f"UMLS installed dir must contain MRCONSO.RRF and MRSTY.RRF: {d}"
            )
    
    def _is_quickumls_index_dir(self, d: str) -> bool:
        """Check if directory contains QuickUMLS index structure."""
        return (os.path.isdir(d) and 
                (os.path.isfile(os.path.join(d, 'metadata.json')) or 
                 os.path.isdir(os.path.join(d, 'umls-simstring.db')) or 
                 os.path.isdir(os.path.join(d, 'index')) or 
                 os.path.isdir(os.path.join(d, 'cui')) or 
                 os.path.isdir(os.path.join(d, 'cui-semtypes.db'))))
    
    def _ensure_quickumls_index(self, umls_dir: str, index_dir: str, build_if_missing: bool):
        """Make or validate QuickUMLS index."""
        if self._is_quickumls_index_dir(index_dir):
            return
        if not build_if_missing:
            raise FileNotFoundError(
                f"QuickUMLS index not found at {index_dir}. "
                f"Set build_index_if_missing=True to create it."
            )
        
        os.makedirs(index_dir, exist_ok=True)
        # Build via CLI: python -m quickumls.install <UMLS> <INDEX>
        cmd = ["python", "-m", "quickumls.install", umls_dir, index_dir]
        print(f"[INFO] Building QuickUMLS index...\n$ {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        if not self._is_quickumls_index_dir(index_dir):
            raise RuntimeError("QuickUMLS index build did not produce expected structure.")
    
    def _create_extractor(self):
        """Create and return QuickUMLS extractor."""
        return QuickUMLSExtractor(
            index_dir=self.quickumls_index,
            threshold=self.threshold,
            window=self.window,
            semtypes=self.semtypes,
            matching=self.matching
        )
    
    def __call__(self, refs: List[str], hyps: List[str]) -> List[float]:
        """
        Calculate MEDCON F1 scores for reference-hypothesis pairs.
        
        Args:
            refs: List of reference texts
            hyps: List of hypothesis/prediction texts
            
        Returns:
            List of F1 scores for each pair
        """
#        if len(refs) != len(hyps):
#            raise ValueError(f"Length mismatch: refs={len(refs)}, hyps={len(hyps)}")
        
        # Convert to strings and handle None values
        refs = [str(r) if r is not None else "" for r in refs]
        hyps = [str(h) if h is not None else "" for h in hyps]
        
        # Calculate scores using original MEDCON method
        scores = [
            self._umls_score_individual_like_original(ref, hyp)
            for ref, hyp in zip(refs, hyps)
        ]
        
        return scores
    
    def _umls_score_individual_like_original(self, reference: str, prediction: str) -> float:
        """Calculate F1 score using original MEDCON method."""
        true_concept, true_cuis = self.extractor.get_matches_like_original(reference)
        pred_concept, pred_cuis = self.extractor.get_matches_like_original(prediction)
        
        try:
            num_t = 0
            for key in true_concept:
                for cui in true_concept[key]:
                    if cui in pred_cuis:
                        num_t += 1
                        break
            
            precision = num_t * 1.0 / max(len(pred_concept.keys()), 1)
            recall = num_t * 1.0 / max(len(true_concept.keys()), 1)
            
            if (precision + recall) == 0:
                return 0.0
            else:
                return 2 * precision * recall / (precision + recall)
        except Exception:
            return 0.0
    
    def get_mean_score(self, refs: List[str], hyps: List[str]) -> float:
        """Get mean F1 score for all pairs."""
        scores = self(refs, hyps)
        return float(np.mean(scores)) if scores else 0.0


class QuickUMLSExtractor:
    """QuickUMLS-based concept extractor."""
    
    def __init__(self, index_dir: str, threshold: float = 0.5, window: int = 5,
                 semtypes: Optional[List[str]] = None, matching: bool = False):
        from quickumls import QuickUMLS
        
        self.matching = matching
        
        kwargs = {
            "threshold": threshold,
            "window": window
        }
        
        # Try to use SEMANTICS first, then fall back to semtypes parameter
        try:
            from semantics import SEMANTICS
            kwargs["accepted_semtypes"] = set(SEMANTICS)
        except Exception:
            if semtypes:  # Use CLI-provided semtypes only if SEMANTICS import fails
                kwargs["accepted_semtypes"] = set(semtypes)
        
        self.matcher = QuickUMLS(index_dir, **kwargs)
        self.ignore_syntax = True     # Same as original
        self.best_match = False       # Same as original
    
    def get_matches_like_original(self, text: str):
        """Get matches using original method logic."""
        concepts = {}
        cui_list = []
        
        if not text:
            if self.matching:
                print("[matching] Empty text input")
            return concepts, cui_list
        
        if self.matching:
            print(f"[matching] Processing text: '{text[:100]}...'")
        
        try:
            matches = self.matcher.match(text, ignore_syntax=True)  # best_match=False
            if self.matching:
                print(f"[matching] Raw matches found: {len(matches)} groups")
            
            for group_idx, group in enumerate(matches):
                if self.matching:
                    print(f"[matching] Group {group_idx}: {len(group)} matches")
                for m_idx, m in enumerate(group):
                    term = m.get('term')
                    cui = m.get('cui')
                    similarity = m.get('similarity', 'N/A')
                    if self.matching:
                        print(f"[matching]   Match {m_idx}: term='{term}', cui='{cui}', sim={similarity}")
                    
                    if term and cui:
                        if cui not in concepts.get(term, []):
                            concepts[term] = concepts.get(term, []) + [cui]
                            cui_list.append(cui)
            
            if self.matching:
                print(f"[matching] Final concepts: {concepts}")
                print(f"[matching] Final CUI list: {cui_list}")
            
        except Exception as e:
            if self.matching:
                print(f"[matching] Exception in matching: {e}")
                import traceback
                traceback.print_exc()
        
        return concepts, cui_list
    
    def extract_concepts(self, text: str) -> Set[str]:
        """Extract concept CUIs from text."""
        if not text:
            return set()
        
        cuis = set()
        matches = self.matcher.match(text, ignore_syntax=self.ignore_syntax, best_match=self.best_match)
        
        for spans in matches:
            for m in spans:
                c = m.get("cui")
                if c:
                    cuis.add(c)
        
        return cuis


# Example usage and test
if __name__ == "__main__":
    # Example medical texts
    refs = [
        "heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
        "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
        "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
        "Ms. ___ is a ___ woman with a history of T2DM, PVD \ns/p R BKA, ESRD on HD, and CAD s/p DES to LCx ___ and ___, \nrecently admitted several time for chronic chest pain, who \npresents from ___ for hypoxemia, dyspnea, \nand left sided chest pain.  \n\n# NSTEMI:  Patient has known 3 vessel disease (LM distal 60% \nstenosis, LCX ostial 90% stenosis before stent mid 60% stenosis, \nLAD diffuse disease segmental mid 60% stenosis, RCA mid ___ \nstenosis, L to distal R collaterals).  Cardiac surgery \npreviously deemed her not to be a candidate for CABG due to poor \nvenous conduits.  During this admission, patient presented with \nchest pain and nausea/vomiting. Trop-T trended 0.38 -> 0.39 -> \n0.39 -> 0.39, which is actually below her recent baseline of \n0.50.  EKG showed ST depressions in lateral leads.  Patient was \ntreated with heparin gtt, atorvastatin, metoprolol, ranolazine, \nASA, clopidogrel, and Imdur.  Overnight on ___, she had more \nchest pain that did not resolve with SL nitro.  She was put on a \nnitroglycerin drip and her blood pressure decreased to ___ \nsystolic.  She became hypoxic, which resolved on 2L O2.  The \nnext morning, she was evaluated by Dr. ___ high-risk PCI.  \nPatient underwent catheterization on ___ and received one \nDES to LAD with angioplasty (no new stent) in LCx.  \n\n# Hypotension:  Cardiac catheterization was complicated by chest \npain and hypotension, thought to be due to a plaque shift into \nthe LAD.  She was monitored in the CCU and was briefly on \ndopamine.  She continued to have persistent hypotension, which \nwas thought to be multifactorial (cardiac, post sedatives, poor \npo intake).  She was started on midodrine in the CCU but this \nwas discontinued on arrival to the floor.  Home amlodipine and \nisosorbide mononitrate were held and discontinued on discharge.  \nHer metoprolol dose was decreased and she was ultimately \ndischarged on metoprolol XL 100 mg daily. \n\n# Acute on chronic diastolic CHF c/b ESRD on HD:  Patient had \nevidence of fluid overload on admission by CXR, physical exam, \nand BNP of > 20000.  Patient did not respond to 80 mg IV Lasix \n(she is primarily anuric).  Patient was treated with dialysis.  \nShe may benefit from a sleep study as she is reporting dyspnea \nwhile lying flat (may be related to OSA rather than heart \nfailure).\n\n# ESRD on HD:  Patient was continued on sevelamer, nephrocaps, \nand cinacalcet.  Multivitamins were discontinued per pharmacy \n(should not be used with nephrocaps).  She was continued on \ndialysis ___ schedule).  On first dialysis session in-house \n(___), limited volume was removed during HD because of \nhypotension. Her anti-hypertensive regimen was modified as \nabove.\n\n# Diabetes type 2: C/b neuropathy, nephropathy, and likely \ngastroparesis.  Lantus was decreased to 50 units to 44 units qhs \ndue to low fasting blood sugars.  HISS was continued.  Patient \nwould benefit from outpatient work-up for gastroparesis."
    ]
    
    hyps = [
        "there are moderate bilateral pleural effusions with overlying atelectasis, underlying consolidation not excluded. mild prominence of the interstitial markings suggests mild pulmonary edema. the cardiac silhouette is mildly enlarged. the mediastinal contours are unremarkable. there is no evidence of pneumothorax.",
        "there are moderate bilateral pleural effusions with overlying atelectasis, underlying consolidation not excluded. mild prominence of the interstitial markings suggests mild pulmonary edema. the cardiac silhouette is mildly enlarged. the mediastinal contours are unremarkable. there is no evidence of pneumotharax.",
        "there are moderate bilateral pleural effusions with overlying atelectasis, underlying consolidation not excluded. mild prominence of the interstitial markings suggests mild pulmonary edema. the cardiac silhouette is mildly enlarged. the mediastinal contours are unremarkable. there is no evidence of pneumothorax.",
        "Ms. ___ is a ___ woman with a history of T2DM, PVD \ns/p R BKA, ESRD on HD, and CAD s/p DES to LCx ___ and ___, \nrecently admitted several time for chronic chest pain, who \npresents from ___ for hypoxemia, dyspnea, \nand left sided chest pain.  \n\n# NSTEMI:  Patient has known 3 vessel disease (LM distal 60% \nstenosis, LCX ostial 90% stenosis before stent mid 60% stenosis, \nLAD diffuse disease segmental mid 60% stenosis, RCA mid ___ \nstenosis, L to distal R collaterals).  Cardiac surgery \npreviously deemed her not to be a candidate for CABG due to poor \nvenous conduits.  During this admission, patient presented with \nchest pain and nausea/vomiting. Trop-T trended 0.38 -> 0.39 -> \n0.39 -> 0.39, which is actually below her recent baseline of \n0.50.  EKG showed ST depressions in lateral leads.  Patient was \ntreated with heparin gtt, atorvastatin, metoprolol, ranolazine, \nASA, clopidogrel, and Imdur.  Overnight on ___, she had more \nchest pain that did not resolve with SL nitro.  She was put on a \nnitroglycerin drip and his blood pressure decreased to ___ \nsystolic.  She became hypoxic, which resolved on 2L O2.  The \nnext morning, she was evaluated by Dr. ___ high-risk PCI.  \nPatient underwent catheterization on ___ and received one \nDES to LAD with angioplasty (no new stent) in LCx.  \n\n# Hypotension:  Cardiac catheterization was complicated by chest \npain and hypotension, thought to be due to a plaque shift into \nthe LAD.  She was monitored in the CCU and was briefly on \ndopamine.  She continued to have persistent hypotension, which \nwas thought to be multifactorial (cardiac, post sedatives, poor \npo intake).  She was started on midodrine in the CCU but this \nwas discontinued on arrival to the floor.  Home amlodipine and \nisosorbide mononitrate were held and discontinued on discharge.  \nHer metoprolol dose was decreased and she was ultimately \ndischarged on metoprolol XL 100 mg daily. \n\n# Acute on chronic diastolic CHF c/b ESRD on HD:  Patient had \nevidence of fluid overload on admission by CXR, physical exam, \nand BNP of > 20000.  Patient did not respond to 80 mg IV Lasix \n(she is primarily anuric).  Patient was treated with dialysis.  \nShe may benefit from a sleep study as she is reporting dyspnea \nwhile lying flat (may be related to OSA rather than heart \nfailure).\n\n# ESRD on HD:  Patient was continued on sevelamer, nephrocaps, \nand cinacalcet.  Multivitamins were discontinued per pharmacy \n(should not be used with nephrocaps).  She was continued on \ndialysis ___ schedule).  On first dialysis session in-house \n(___), limited volume was removed during HD because of \nhypotension. Her anti-hypertensive regimen was modified as \nabove.\n\n# Diabetes type 2: C/b neuropathy, nephropathy, and likely \ngastroparesis.  Lantus was decreased to 50 units to 44 units qhs \ndue to low fasting blood sugars.  HISS was continued.  Patient \nwould benefit from outpatient work-up for gastroparesis."
    ]
    
    # Initialize scorer (you need to set the actual paths)
    scorer = MedconScore(
        umls_installed_dir="./2025AA",
        quickumls_index="quickumls",
        threshold=0.5,
        window=5,
        build_index_if_missing=True,
        matching=False  # Enable matching debug output
    )
    
    # Calculate scores
    scores = scorer(refs=refs, hyps=hyps)
    mean_score = scorer.get_mean_score(refs=refs, hyps=hyps)
    
    print("Individual scores:", scores)
    print("Mean score:", mean_score)
