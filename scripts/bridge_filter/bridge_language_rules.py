"""Text rules for Bridge instructions.

Normalize ``language_instruction``, flag empty instructions and annotation
comments as invalid, and mark uncertain wording for review. Task queries use
entity, action, and relation rules to identify strict, broad, and partial matches.

The scanner in ``scripts/prepare_bridge.py`` reads TFRecords and writes the
results. Keeping I/O in the scanner lets these rules run in isolation in tests
and on instructions from other dataset formats.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u0060": "'"})
_NON_TEXT = re.compile(r"[^a-z0-9'\s]+")
_SPACE = re.compile(r"\s+")


def normalize_instruction(text: str) -> str:
    """Normalize English instructions while preserving directional words.

    Standardize Unicode, case, separators, punctuation, and whitespace. Keep
    words such as ``on``, ``off``, ``from``, and ``in`` because they specify direction.
    """
    text = unicodedata.normalize("NFKC", text or "").translate(_APOSTROPHES)
    text = text.replace("_", "-").lower()
    text = re.sub(r"[-/]", " ", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = normalize_instruction(phrase).split()
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, words)) + r"(?![a-z0-9])")


@dataclass(frozen=True)
class Hit:
    """A lexicon phrase and its span in the normalized instruction."""
    category: str
    phrase: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "phrase": self.phrase,
            "start": self.start,
            "end": self.end,
        }


class PhraseLexicon:
    """Compile YAML phrase categories into reusable regular expressions.

    Prefer longer phrases when matches overlap, such as ``pick up`` and ``pick``.
    """

    def __init__(self, categories: Mapping[str, Sequence[str]]):
        self._entries: list[tuple[str, str, re.Pattern[str]]] = []
        for category, phrases in categories.items():
            for phrase in sorted(set(phrases), key=lambda value: (-len(value), value)):
                self._entries.append((category, normalize_instruction(phrase), _phrase_pattern(phrase)))

    def find(self, text: str) -> list[Hit]:
        candidates: list[Hit] = []
        for category, phrase, pattern in self._entries:
            for match in pattern.finditer(text):
                candidates.append(Hit(category, phrase, match.start(), match.end()))

        # Prefer the longest phrase when entries overlap, e.g. "pick up" over "pick".
        candidates.sort(key=lambda hit: (hit.start, -(hit.end - hit.start), hit.category))
        kept: list[Hit] = []
        for hit in candidates:
            if any(not (hit.end <= old.start or hit.start >= old.end) for old in kept):
                continue
            kept.append(hit)
        return sorted(kept, key=lambda hit: (hit.start, hit.end))


def _term_hits(text: str, terms: Iterable[str], category: str) -> list[Hit]:
    """Find entity terms from the task configuration, such as carrot and plate."""
    hits: list[Hit] = []
    for term in sorted(set(terms), key=lambda value: (-len(value), value)):
        phrase = normalize_instruction(term)
        for match in _phrase_pattern(phrase).finditer(text):
            hits.append(Hit(category, phrase, match.start(), match.end()))
    return sorted(hits, key=lambda hit: (hit.start, -(hit.end - hit.start)))


def _split_clauses(text: str, actions: PhraseLexicon) -> list[str]:
    """Split sequences of actions while keeping phrases such as ``between X and Y`` intact."""
    rough = [part.strip() for part in re.split(r"\s*;\s*|\b(?:and then|then)\b", text) if part.strip()]
    clauses: list[str] = []
    for part in rough:
        cursor = 0
        for match in re.finditer(r"\band\b", part):
            right = part[match.end() :].strip()
            # Split plain "and" only when the right side starts a new action soon.
            right_hits = actions.find(right)
            if right_hits and right_hits[0].start <= 12:
                left = part[cursor : match.start()].strip()
                if left:
                    clauses.append(left)
                cursor = match.end()
        tail = part[cursor:].strip()
        if tail:
            clauses.append(tail)
    return clauses or [text]


class LanguageAnalyzer:
    """Assess instruction validity, extract semantic slots, and match task queries."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.actions = PhraseLexicon(config["actions"])
        self.relations = PhraseLexicon(config["relations"])
        self.meta_patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in config.get("validity", {}).get("meta_comment_patterns", [])
        ]

    def analyze(self, raw_text: str) -> dict[str, Any]:
        """Assess instruction validity and extract actions, relations, and clauses.

        - ``valid`` indicates a recognized action with no quality issues.
        - ``invalid`` indicates empty text or a recognized annotation comment.
        - ``review`` flags unfamiliar action wording or a possible encoding issue.
        """
        text = normalize_instruction(raw_text)
        reasons: list[str] = []

        if not text:
            validity = "invalid"
            quality = "empty"
            reasons.append("empty_instruction")
        else:
            meta_matches = [pattern.pattern for pattern in self.meta_patterns if pattern.search(text)]
            if meta_matches:
                validity = "invalid"
                quality = "meta_comment"
                reasons.append("meta_comment_pattern")
            elif "\ufffd" in raw_text:
                validity = "review"
                quality = "malformed_encoding"
                reasons.append("unicode_replacement_character")
            else:
                validity = "valid"
                quality = "task_instruction"

        action_hits = self.actions.find(text)
        relation_hits = self.relations.find(text)
        clauses = _split_clauses(text, self.actions) if text else []

        if validity == "valid" and not action_hits:
            # Do not silently discard terse but possibly valid labels such as
            # "lever vertical to front". They require review.
            validity = "review"
            quality = "unknown_action_wording"
            reasons.append("no_known_action_phrase")

        return {
            "raw": raw_text,
            "normalized": text,
            "validity": validity,
            "quality": quality,
            "reasons": reasons,
            "actions": [hit.as_dict() for hit in action_hits],
            "relations": [hit.as_dict() for hit in relation_hits],
            "clauses": clauses,
            "multi_step": len(clauses) > 1,
        }

    def match_query(self, analysis: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
        """Match an analyzed instruction against a named task configuration.

        - A ``strict`` match requires an action, object, relation, and counterpart
          in that order within one clause. The action and relation must agree on
          direction.
        - A ``broad`` match contains both entities, with ambiguous roles due to
          pronouns, reversed word order, or other wording.
        - The ``partial_candidate`` flag marks instructions containing both
          entities with insufficient action or relation evidence for a match.
        - A ``none`` match level means no strict or broad match was found.

        A task can group mutually exclusive subtasks under ``alternatives``.
        For example, ``simpler_pick_place`` groups four pick-and-place tasks.
        Checking each entity pair separately keeps the spoon paired with the
        towel and the carrot with the plate.
        """
        alternatives = query.get("alternatives", [])
        if alternatives:
            subtask_results = {
                alternative["subtask_name"]: self.match_query(analysis, alternative)
                for alternative in alternatives
            }
            strict_subtasks = [
                name for name, result in subtask_results.items() if result["strict"]
            ]
            candidate_subtasks = [
                name for name, result in subtask_results.items() if result["candidate"]
            ]
            review_subtasks = [
                name for name, result in subtask_results.items() if result["review_recommended"]
            ]
            partial_subtasks = [
                name for name, result in subtask_results.items() if result["partial_candidate"]
            ]

            if strict_subtasks:
                level = "strict"
                evidence_names = strict_subtasks
            elif candidate_subtasks:
                level = "broad"
                evidence_names = candidate_subtasks
            else:
                level = "none"
                evidence_names = partial_subtasks

            evidence = [
                {
                    "subtask_name": name,
                    "match_level": subtask_results[name]["match_level"],
                    "evidence": subtask_results[name]["evidence"],
                }
                for name in evidence_names
            ]
            return {
                "candidate": bool(candidate_subtasks),
                "strict": bool(strict_subtasks),
                "partial_candidate": not candidate_subtasks and bool(partial_subtasks),
                "review_recommended": not strict_subtasks
                and bool(review_subtasks or partial_subtasks),
                "match_level": level,
                "directions": sorted(
                    {
                        direction
                        for name in evidence_names
                        for direction in subtask_results[name]["directions"]
                    }
                ),
                "evidence": evidence,
                "matched_subtasks": strict_subtasks or candidate_subtasks,
                "review_subtasks": review_subtasks or partial_subtasks,
                "subtask_results": subtask_results,
                "term_presence": {},
                "missing_slots": [],
            }

        if query.get("selection_mode") == "valid_object":
            text = str(analysis["normalized"])
            object_hits = _term_hits(text, query["object_terms"], "object")
            allowed_action_phrases = {
                normalize_instruction(value) for value in query.get("allowed_action_phrases", [])
            }
            action_hits = [
                hit
                for hit in self.actions.find(text)
                if not allowed_action_phrases or hit.phrase in allowed_action_phrases
            ]
            relation_hits = self.relations.find(text)

            # Require eggplant to occur before the first relation that follows
            # an action. This rejects sentences such as "put the spoon under
            # the eggplant", where eggplant is the reference object, while
            # retaining coordinate targets such as "move eggplant to the left".
            subject_objects = [
                obj
                for obj in object_hits
                if any(
                    action.start < obj.start
                    and not any(action.start < relation.start < obj.start for relation in relation_hits)
                    for action in action_hits
                )
            ]
            selected = analysis["validity"] == "valid" and bool(subject_objects) and bool(action_hits)
            evidence = [
                {
                    "object": obj.as_dict(),
                    "actions": [action.as_dict() for action in action_hits],
                    "selection_basis": "valid_language_and_subject_object",
                }
                for obj in subject_objects
            ]
            term_presence = {
                "object": bool(object_hits),
                "action": bool(action_hits),
            }
            return {
                "candidate": selected,
                "strict": selected,
                "partial_candidate": False,
                "review_recommended": False,
                "match_level": "strict" if selected else "none",
                "directions": [],
                "evidence": evidence,
                "term_presence": term_presence,
                "missing_slots": [name for name, present in term_presence.items() if not present],
            }

        text = str(analysis["normalized"])
        clauses = list(analysis.get("clauses") or ([text] if text else []))
        object_terms = query["object_terms"]
        counterpart_terms = query["counterpart_terms"]

        toward_actions = set(query.get("toward_action_categories", []))
        toward_relations = set(query.get("toward_relation_categories", []))
        away_actions = set(query.get("away_action_categories", []))
        away_relations = set(query.get("away_relation_categories", []))
        broad_relations = set(query.get("broad_relation_categories", []))
        allowed_action_phrases = {
            normalize_instruction(value) for value in query.get("allowed_action_phrases", [])
        }
        allowed_relation_phrases = {
            normalize_instruction(value) for value in query.get("allowed_relation_phrases", [])
        }

        def query_actions(clause: str) -> list[Hit]:
            """Apply the task's allowed action phrases. An empty list allows all actions."""
            hits = self.actions.find(clause)
            return [hit for hit in hits if not allowed_action_phrases or hit.phrase in allowed_action_phrases]

        def query_relations(clause: str) -> list[Hit]:
            """Apply the task's allowed relation phrases. Relation categories determine direction."""
            hits = self.relations.find(clause)
            return [
                hit for hit in hits if not allowed_relation_phrases or hit.phrase in allowed_relation_phrases
            ]

        strict_evidence: list[dict[str, Any]] = []
        broad_evidence: list[dict[str, Any]] = []
        object_seen_before = False

        for clause_index, clause in enumerate(clauses):
            objects = _term_hits(clause, object_terms, "object")
            counterparts = _term_hits(clause, counterpart_terms, "counterpart")
            actions = query_actions(clause)
            relations = query_relations(clause)

            for action in actions:
                for obj in objects:
                    for relation in relations:
                        for counterpart in counterparts:
                            canonical_order = action.start < obj.start < relation.start < counterpart.start
                            direction = None
                            if action.category in toward_actions and relation.category in toward_relations:
                                direction = "toward"
                            elif action.category in away_actions and relation.category in away_relations:
                                direction = "away"

                            evidence = {
                                "clause_index": clause_index,
                                "clause": clause,
                                "direction": direction or "ambiguous",
                                "action": action.as_dict(),
                                "object": obj.as_dict(),
                                "relation": relation.as_dict(),
                                "counterpart": counterpart.as_dict(),
                                "canonical_order": canonical_order,
                            }
                            if canonical_order and direction:
                                strict_evidence.append(evidence)
                            elif relation.category in broad_relations:
                                broad_evidence.append(evidence)

            # Common coreference: "take the carrot ... and put it on the plate".
            if object_seen_before and counterparts:
                pronoun = re.search(r"\b(?:it|them|this object|the object)\b", clause)
                for action in actions:
                    for relation in relations:
                        for counterpart in counterparts:
                            direction = None
                            if action.category in toward_actions and relation.category in toward_relations:
                                direction = "toward"
                            elif action.category in away_actions and relation.category in away_relations:
                                direction = "away"
                            if direction and action.start < relation.start < counterpart.start:
                                broad_evidence.append(
                                    {
                                        "clause_index": clause_index,
                                        "clause": clause,
                                        "direction": direction,
                                        "coreference": "explicit_pronoun" if pronoun else "implicit_carryover",
                                        "action": action.as_dict(),
                                        "relation": relation.as_dict(),
                                        "counterpart": counterpart.as_dict(),
                                        "canonical_order": False,
                                    }
                                )
            object_seen_before = object_seen_before or bool(objects)

        all_objects = _term_hits(text, object_terms, "object")
        all_counterparts = _term_hits(text, counterpart_terms, "counterpart")
        all_actions = query_actions(text)
        all_relations = [hit for hit in query_relations(text) if hit.category in broad_relations]

        if strict_evidence:
            level = "strict"
            evidence = strict_evidence
        elif broad_evidence:
            level = "broad"
            evidence = broad_evidence
        elif all_objects and all_counterparts and all_actions and all_relations:
            # All ingredients exist but semantic roles/order were not established.
            level = "broad"
            evidence = [{"reason": "terms_present_but_roles_ambiguous"}]
        else:
            level = "none"
            evidence = []

        term_presence = {
            "object": bool(all_objects),
            "counterpart": bool(all_counterparts),
            "action": bool(all_actions),
            "relation": bool(all_relations),
        }
        partial_candidate = term_presence["object"] and term_presence["counterpart"] and level == "none"

        return {
            "candidate": level != "none",
            "strict": level == "strict",
            "partial_candidate": partial_candidate,
            "review_recommended": level == "broad" or partial_candidate,
            "match_level": level,
            "directions": sorted(
                {item.get("direction", "ambiguous") for item in evidence if isinstance(item, dict)}
            ),
            "evidence": evidence,
            "term_presence": term_presence,
            "missing_slots": [name for name, present in term_presence.items() if not present],
        }

    def analyze_with_queries(self, raw_text: str) -> dict[str, Any]:
        """Analyze the instruction once and run each configured task query on the result."""
        analysis = self.analyze(raw_text)
        analysis["queries"] = {
            name: self.match_query(analysis, query)
            for name, query in self.config.get("queries", {}).items()
        }
        return analysis
