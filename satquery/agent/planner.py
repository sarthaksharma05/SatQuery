"""Query interpretation and workflow planning.

``RulePlanner`` is a deterministic planner: it classifies the task and proposes a
tool sequence with permitted parameters. The proposal is passed through
``validate_plan`` by the controller.
"""
from __future__ import annotations

import json
import re
from typing import Any

from satquery.agent.plan import ALLOWED_TASKS, INTENTS, Plan, PlanStep
from satquery.config import settings
from satquery.imagery.raster import Modality
from satquery.imagery.validate import InputConfig, ValidationReport
from satquery.tools.base import TaskType
from satquery.tools.registry import Registry

_LOCATE = re.compile(r"\b(highlight|locate|localize|localise|where is|where are|find|mark|outline|segment|box|"
                     r"bounding|point out|show me|delineate|identify (the|all)?\s*\w*\s*(regions?|areas?|zones?)|"
                     r"identify .* regions|map the)\b")
_DESCRIBE = re.compile(r"\b(describe|description|caption|summari[sz]e|overview|what (can you see|is visible|is in)|"
                       r"tell me about|what changed|what has changed|what are the changes|explain)\b")


def detect_intent(query: str, config: InputConfig) -> str:
    q = query.lower()
    if config is InputConfig.BITEMPORAL:
        return "describe" if (_DESCRIBE.search(q) or re.search(r"\bwhere\b", q)) and not re.match(
            r"^\s*(is|are|has|have|did|does|do)\b", q) else "qa"
    if _LOCATE.search(q):
        return "locate"
    if _DESCRIBE.search(q):
        return "describe"
    return "qa"


class RulePlanner:
    name = "rules"

    def plan(self, query: str, report: ValidationReport, registry: Registry) -> Plan:
        cfg = report.config
        intent = detect_intent(query, cfg)
        vlm_ok = registry.available("rs_vlm")[0]
        scene_ok = registry.available("rs_scene_classifier")[0]
        rsvqa_ok = registry.available("rsvqa_vqa")[0]
        hr_ok = registry.available("hr_vqa")[0]
        optical = any(r.modality.is_optical for r in report.rasters)
        steps: list[PlanStep] = []
        if cfg is InputConfig.SINGLE:
            lc = "landcover_sar" if report.rasters[0].modality is Modality.SAR else "landcover_optical"
            if intent == "locate":
                task = TaskType.GROUNDING
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "grounding"}, "referring-expression grounding (boxes)"))
                steps += [PlanStep(lc, {}, "land-cover masks as grounding evidence"),
                          PlanStep("region_grounder", {}, "select regions matching the referred class")]
                if registry.available("hr_grounder")[0] and lc == "landcover_optical":
                    steps.append(PlanStep("hr_grounder", {}, "approximate box for a referred object (high-resolution images)"))
            elif intent == "describe":
                task = TaskType.CAPTION
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "caption"}, "scene caption"))
                steps += [PlanStep(lc, {}, "quantitative land-cover evidence"),
                          PlanStep("landcover_reasoner", {"mode": "describe"}, "land-cover scene description")]
                if hr_ok and optical:
                    steps.append(PlanStep("hr_vqa", {"mode": "describe"}, "object-level description (high-resolution images)"))
                if scene_ok and optical:
                    steps.append(PlanStep("rs_scene_classifier", {"mode": "describe"}, "EuroSAT-adapted land-use classification"))
            else:
                task = TaskType.VQA
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "vqa"}, "answer the question"))
                steps += [PlanStep(lc, {}, "land-cover evidence"),
                          PlanStep("landcover_reasoner", {"mode": "qa"}, "evidence-based answer / cross-check")]
                if scene_ok and optical:
                    steps.append(PlanStep("rs_scene_classifier", {"mode": "qa"}, "land-use answer from the adapted classifier"))
                if rsvqa_ok and optical:
                    steps.append(PlanStep("rsvqa_vqa", {}, "answer from the RSVQA-trained VQA specialist"))
                if hr_ok and optical:
                    steps.append(PlanStep("hr_vqa", {}, "answer from the VRSBench-trained high-resolution specialist"))
        elif cfg is InputConfig.CROSS_MODAL:
            task = TaskType.CROSS_MODAL
            steps = [PlanStep("landcover_optical", {}, "optical/multispectral land cover + cloud mask"),
                     PlanStep("landcover_sar", {}, "SAR land cover (cloud-independent)"),
                     PlanStep("optical_sar_fusion", {}, "fuse complementary optical and SAR evidence")]
            if intent == "locate" or re.search(r"\b(regions?|areas?)\b", query.lower()):
                steps.append(PlanStep("region_grounder", {}, "highlight the requested classes on the fused map"))
            steps.append(PlanStep("landcover_reasoner", {"mode": "qa" if intent == "qa" else "describe"},
                                  "answer from fused evidence"))
            if scene_ok:
                steps.append(PlanStep("rs_scene_classifier", {"mode": "qa" if intent == "qa" else "describe"},
                                      "land use of the optical image (EuroSAT-adapted)"))
            if vlm_ok:
                steps.append(PlanStep("rs_vlm", {"mode": "caption" if intent != "qa" else "vqa"},
                                      "joint optical+SAR reading by the RS-adapted VLM"))
        else:
            task = TaskType.CHANGE_DESCRIPTION if intent == "describe" else TaskType.CHANGE_VQA
            steps = [PlanStep("change_detector", {}, "where did it change"),
                     PlanStep("change_semantics", {}, "what changed into what"),
                     PlanStep("change_qa", {"mode": "describe" if intent == "describe" else "auto"},
                              "answer / describe the change")]
            if intent != "describe" and registry.available("cdvqa_vqa")[0] and all(
                    r.modality.is_optical for r in report.rasters):
                steps.append(PlanStep("cdvqa_vqa", {}, "answer from the CDVQA-trained change specialist"))
            if vlm_ok and intent == "describe":
                steps.append(PlanStep("rs_vlm", {}, "auxiliary visual change description"))
        return Plan(task=task, intent=intent, steps=steps, planner=self.name,
                    rationale=f"rule-based: {cfg.value} input, intent '{intent}'")


class PlannerError(RuntimeError):
    pass
