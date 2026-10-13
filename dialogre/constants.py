
import json
import logging
import os

logger = logging.getLogger(__name__)

DIALOGRE_RELATIONS = [
    "per:positive_impression",
    "per:negative_impression",
    "per:acquaintance",
    "per:alumni",
    "per:boss",
    "per:subordinate",
    "per:client",
    "per:dates",
    "per:friends",
    "per:girl/boyfriend",
    "per:neighbor",
    "per:roommate",
    "per:children",
    "per:other_family",
    "per:parents",
    "per:siblings",
    "per:spouse",
    "per:place_of_residence",
    "per:place_of_birth",
    "per:visited_place",
    "per:origin",
    "per:employee_or_member_of",
    "per:schools_attended",
    "per:works",
    "per:age",
    "per:date_of_birth",
    "per:major",
    "per:place_of_work",
    "per:title",
    "per:alternate_names",
    "per:pet",
    "gpe:residents_of_place",
    "gpe:births_in_place",
    "gpe:visitors_of_place",
    "org:employees_or_members",
    "org:students",
]

DIALOGRE_REL2ID = {rel: idx + 1 for idx, rel in enumerate(DIALOGRE_RELATIONS)}
DIALOGRE_REL2ID["no_relation"] = 0
DIALOGRE_ID2REL = {idx: rel for rel, idx in DIALOGRE_REL2ID.items()}


def load_relation_schema(data_dir):
    candidates = ["meld_rel2id.json", "dialogre_rel2id.json"]
    for name in candidates:
        schema_path = os.path.join(data_dir, name)
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                return schema
            except Exception as e:
                logger.warning(f"Failed to load schema from {schema_path}: {e}")

    return {
        "label2id": DIALOGRE_REL2ID,
        "id2label": DIALOGRE_ID2REL,
        "num_relations": 36,
        "num_classes": 37,
    }


def normalize_relation_label(label, label_keys):
    if not label:
        return None

    label = str(label).strip()

    if label in label_keys:
        return label

    for key in label_keys:
        if label.lower() == key.lower():
            return key

    label_no_space = label.replace(" ", "_").lower()
    for key in label_keys:
        key_no_space = key.replace(" ", "_").lower()
        if label_no_space == key_no_space:
            return key

    return None


def get_relation_name(rel_id):
    return DIALOGRE_ID2REL.get(rel_id, "unknown")


def get_relation_id(rel_name):
    return DIALOGRE_REL2ID.get(rel_name, 0)
