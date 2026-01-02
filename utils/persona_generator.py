import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# 1. Constants & Distributions
# =============================================================================

GENDER_DISTRIBUTION = {"Male": 0.54, "Female": 0.46}

AGE_DISTRIBUTION = {
    "18-19": 0.04, "20-29": 0.24, "30-39": 0.26,
    "40-49": 0.21, "50-59": 0.17, "60+": 0.08
}

GAMER_TYPE_DISTRIBUTION = {
    "ultimate_gamer": 0.13, "all_round_enthusiast": 0.09, "cloud_gamer": 0.19,
    "conventional_player": 0.04, "hardware_enthusiast": 0.09, "popcorn_gamer": 0.13,
    "backseat_gamer": 0.06, "time_filler": 0.27
}

GAMER_TYPES = {
    "ultimate_gamer": {
        "type_name_display": "The Ultimate Gamer",
        "description": "A passionate gamer who spares no time or money on games.",
        "traits": {"spending_level": "Very High", "time_investment": "20+ hours/week", "platform_preference": ["PC", "Console"], "purchase_timing": "Day-1 Purchase", "information_seeking": "Buys regardless of reviews", "brand_loyalty": "Very High"},
        "expected_score_range": (75, 95)
    },
    "all_round_enthusiast": {
        "type_name_display": "The All-Round Enthusiast",
        "description": "Enjoys all genres and seeks a balanced gaming life.",
        "traits": {"spending_level": "Medium-High", "time_investment": "10-15 hours/week", "platform_preference": ["PC", "Console", "Mobile"], "purchase_timing": "Buys after checking reviews", "information_seeking": "Checks reviews carefully", "brand_loyalty": "Medium"},
        "expected_score_range": (50, 80)
    },
    "cloud_gamer": {
        "type_name_display": "The Cloud Gamer",
        "description": "Prefers streaming or budget games without a high-end PC.",
        "traits": {"spending_level": "Low-Medium", "time_investment": "5-10 hours/week", "platform_preference": ["Cloud Gaming"], "purchase_timing": "Buys on deep sale", "information_seeking": "Checks optimization reviews", "brand_loyalty": "Low"},
        "expected_score_range": (20, 60)
    },
    "conventional_player": {
        "type_name_display": "The Conventional Player",
        "description": "Plays only familiar games; uninterested in new releases.",
        "traits": {"spending_level": "Very Low", "time_investment": "5-10 hours/week", "platform_preference": ["PC", "Console"], "purchase_timing": "Rarely buys", "information_seeking": "Indifferent", "brand_loyalty": "N/A"},
        "expected_score_range": (10, 30)
    },
    "hardware_enthusiast": {
        "type_name_display": "The Hardware Enthusiast",
        "description": "Obsessed with latest gear and graphics; buys games for benchmarking.",
        "traits": {"spending_level": "Very High", "time_investment": "15+ hours/week", "platform_preference": ["High-End PC"], "purchase_timing": "Day-1 Purchase", "information_seeking": "Analyzes graphics", "brand_loyalty": "Medium"},
        "expected_score_range": (65, 90)
    },
    "popcorn_gamer": {
        "type_name_display": "The Popcorn Gamer",
        "description": "Enjoys watching Twitch/YouTube more than playing.",
        "traits": {"spending_level": "Very Low", "time_investment": "20+ hours/week (Watching)", "platform_preference": ["YouTube"], "purchase_timing": "Rarely buys", "information_seeking": "Vicarious satisfaction", "brand_loyalty": "N/A"},
        "expected_score_range": (15, 40)
    },
    "backseat_gamer": {
        "type_name_display": "The Backseat Gamer",
        "description": "Used to play hard, now only watches videos due to lack of time.",
        "traits": {"spending_level": "Very Low", "time_investment": "5-10 hours/week (Watching)", "platform_preference": ["YouTube"], "purchase_timing": "Does not buy", "information_seeking": "Nostalgia seeking", "brand_loyalty": "Old Franchise Only"},
        "expected_score_range": (5, 25)
    },
    "time_filler": {
        "type_name_display": "The Time Filler",
        "description": "Plays mobile games only during spare time.",
        "traits": {"spending_level": "Low", "time_investment": "10-15 hours/week", "platform_preference": ["Mobile"], "purchase_timing": "Does not buy", "information_seeking": "Mobile info only", "brand_loyalty": "N/A"},
        "expected_score_range": (0, 20)
    }
}

# =============================================================================
# 2. Generators
# =============================================================================

ENGLISH_SURNAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
ENGLISH_MALE_NAMES = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"]
ENGLISH_FEMALE_NAMES = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]

OCCUPATIONS_BY_AGE = {
    "18-19": ["College Student", "High School Senior", "Gap Year Student", "Part-time Worker"],
    "20-29": ["College Student", "Grad Student", "Entry-level Employee", "Developer", "Freelancer", "YouTuber"],
    "30-39": ["Manager", "CTO", "Designer", "Marketer", "Accountant", "Lawyer"],
    "40-49": ["Director", "Business Owner", "CEO", "Homemaker", "Civil Servant"],
    "50-59": ["Executive", "Business Owner", "Pre-retiree", "Homemaker"],
    "60+": ["Retiree", "Business Owner", "Homemaker"]
}

@dataclass
class Persona:
    id: str
    name: str
    gender: str
    age: int
    age_group: str
    occupation: str
    gamer_type: str
    gamer_type_name_display: str
    traits: Dict
    description: str

def generate_english_name(gender: str) -> str:
    surname = random.choice(ENGLISH_SURNAMES)
    given_name = random.choice(ENGLISH_MALE_NAMES) if gender == "Male" else random.choice(ENGLISH_FEMALE_NAMES)
    return f"{given_name} {surname}"

def sample_age() -> Tuple[str, int]:
    age_group = random.choices(list(AGE_DISTRIBUTION.keys()), weights=list(AGE_DISTRIBUTION.values()))[0]
    ranges = {"18-19": (18, 19), "20-29": (20, 29), "30-39": (30, 39), "40-49": (40, 49), "50-59": (50, 59), "60+": (60, 70)}
    return age_group, random.randint(*ranges[age_group])

def generate_persona(persona_id: str, gamer_type: Optional[str] = None) -> Persona:
    gender = random.choices(["Male", "Female"], weights=[0.54, 0.46])[0]
    age_group, age = sample_age()
    
    if gamer_type is None:
        gamer_type = random.choices(list(GAMER_TYPE_DISTRIBUTION.keys()), weights=list(GAMER_TYPE_DISTRIBUTION.values()))[0]
    
    info = GAMER_TYPES[gamer_type]
    name = generate_english_name(gender)
    occupation = random.choice(OCCUPATIONS_BY_AGE[age_group])
    
    return Persona(
        id=persona_id, 
        name=name, 
        gender=gender, 
        age=age, 
        age_group=age_group, 
        occupation=occupation, 
        gamer_type=gamer_type, 
        gamer_type_name_display=info["type_name_display"], 
        traits=info["traits"],
        description=info["description"]
    )

def generate_balanced_personas(n_per_type: int = 13) -> List[Persona]:
    personas = []
    for gamer_type in GAMER_TYPES.keys():
        for i in range(n_per_type):
            persona_id = f"{gamer_type}_{i+1}"
            personas.append(generate_persona(persona_id, gamer_type=gamer_type))
    return personas

