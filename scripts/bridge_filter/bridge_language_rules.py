"""Bridge 指令的纯文本规则引擎。

这个文件不读取 TFRecord，也不处理图像。它只完成三件事：

1. 规范化 ``language_instruction``；
2. 将空语义、明确的元话语标成 invalid，将无法确认的表达标成 review；
3. 根据任务 YAML 中的实体、动作和关系规则产生 strict/broad/partial 匹配。

TFRecord 扫描和结果落盘位于 ``filter_bridge_task.py``。把规则引擎与数据 I/O
分开，便于用单元测试验证语义规则，也便于未来复用到其他格式的数据集。
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
    """保守地规范化英文指令，同时保留决定方向的单词。

    这里只处理大小写、Unicode、下划线/连字符、标点和多余空格；不会删除
    ``on/off/from/in`` 等对任务方向至关重要的词。
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
    """一个词典短语在规范化指令中的命中位置。"""
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
    """将 YAML 中按类别组织的短语编译为可复用的正则词典。

    当 ``pick`` 和 ``pick up`` 等短语重叠时，优先保留更长的短语。
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
    """查找任务配置中的实体词，例如 carrot、plate。"""
    hits: list[Hit] = []
    for term in sorted(set(terms), key=lambda value: (-len(value), value)):
        phrase = normalize_instruction(term)
        for match in _phrase_pattern(phrase).finditer(text):
            hits.append(Hit(category, phrase, match.start(), match.end()))
    return sorted(hits, key=lambda hit: (hit.start, -(hit.end - hit.start)))


def _split_clauses(text: str, actions: PhraseLexicon) -> list[str]:
    """只切分明确的连续动作子句，避免误切 ``between X and Y``。"""
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
    """执行指令有效性判断、语义槽位抽取和任务查询匹配。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.actions = PhraseLexicon(config["actions"])
        self.relations = PhraseLexicon(config["relations"])
        self.meta_patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in config.get("validity", {}).get("meta_comment_patterns", [])
        ]

    def analyze(self, raw_text: str) -> dict[str, Any]:
        """判断一条指令是否有效，并抽取动作、关系和子句。

        ``validity`` 有三种结果：

        - ``valid``：存在已知动作，且没有命中无效规则；
        - ``invalid``：空语义或高置信度元话语；
        - ``review``：可能有效，但当前词典无法可靠识别动作/编码。

        只有 high-precision 情况会直接 invalid，未知表达不会静默丢弃。
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
        """把一条已分析指令与一个具名任务配置匹配。

        匹配级别：

        - ``strict``：同一子句内满足 action -> object -> relation -> counterpart，
          且动作类别与关系方向一致；
        - ``broad``：实体齐全，但存在代词指代、词序反转或其他角色歧义；
        - ``partial_candidate``：两个实体均出现，但动作/关系词典覆盖不足；
        - ``none``：与该任务无关。

        一个总任务也可以用 ``alternatives`` 包含多个互斥子任务。例如
        ``simpler_pick_place`` 汇总四种 PnP 语义，但仍分别检查 spoon/towel、
        carrot/plate 等实体角色，避免实体列表交叉组合造成误匹配。
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
            """应用任务 YAML 的谓词白名单；空白名单表示使用全部通用谓词。"""
            hits = self.actions.find(clause)
            return [hit for hit in hits if not allowed_action_phrases or hit.phrase in allowed_action_phrases]

        def query_relations(clause: str) -> list[Hit]:
            """应用任务 YAML 的关系词白名单；方向仍由关系类别决定。"""
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
        """一次分析文本，并依次执行当前加载的全部具名任务查询。"""
        analysis = self.analyze(raw_text)
        analysis["queries"] = {
            name: self.match_query(analysis, query)
            for name, query in self.config.get("queries", {}).items()
        }
        return analysis
