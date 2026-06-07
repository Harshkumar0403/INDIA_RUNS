"""
constants.py
============
JD structured parse — four signed sections with explicit polarity.
Career event taxonomy and seniority ladder.

These are the semantic anchors of the entire system.
Separated from config.py because they are conceptual constants,
not tunable parameters.
"""

# ─────────────────────────────────────────────────────────────────
# JD SIGNED SECTIONS
# Parsed from job_description.docx manually.
# Each section has: raw_phrases, polarity, weight.
#
# CRITICAL: Section D (EXPLICIT_NEGATIVE) terms carry NEGATIVE
# polarity. A naive keyword extractor treats "LangChain" as
# positive because it appears in the JD. It is NOT —
# it appears in the rejection section.
# ─────────────────────────────────────────────────────────────────
JD_SECTIONS = {

    # ── A: HARD REQUIREMENTS (+++)
    # Production experience REQUIRED. Not nice-to-have.
    "HARD_REQUIREMENT": {
        "phrases": [
            # Embeddings + retrieval systems
            "production experience with embeddings",
            "embeddings-based retrieval",
            "sentence-transformers", "openai embeddings", "bge", "e5",
            "embedding drift", "index refresh",
            "retrieval-quality regression",
            # Vector databases
            "vector databases", "hybrid search infrastructure",
            "pinecone", "weaviate", "qdrant", "milvus",
            "opensearch", "elasticsearch", "faiss",
            "operational experience",
            # Python quality
            "strong python", "code quality",
            # Evaluation frameworks
            "evaluation frameworks for ranking",
            "ndcg", "mrr", "map",
            "offline-to-online correlation",
            "a/b test interpretation",
            "ranking system",
        ],
        "polarity": "positive",
        "weight":   1.00,
    },

    # ── B: IDEAL PROFILE (++)
    # Strong positive signals but not hard gates.
    "IDEAL_PROFILE": {
        "phrases": [
            "applied ml", "applied ai",
            "product companies",
            "end-to-end ranking", "end-to-end search",
            "end-to-end recommendation",
            "real users", "meaningful scale",
            "hybrid vs dense",
            "offline vs online",
            "when to fine-tune", "when to prompt",
            "systems they actually built",
            "noida", "pune",
        ],
        "polarity": "positive",
        "weight":   0.60,
    },

    # ── C: NICE TO HAVE (+)
    # Small bonus, zero penalty if absent.
    "NICE_TO_HAVE": {
        "phrases": [
            "lora", "qlora", "peft", "llm fine-tuning",
            "learning-to-rank", "xgboost-based",
            "hr-tech", "recruiting tech", "marketplace products",
            "distributed systems", "large-scale inference",
            "open-source contributions",
        ],
        "polarity": "positive",
        "weight":   0.25,
    },

    # ── D: EXPLICIT DO NOT WANT (---)
    # Presence in candidate profile REDUCES score.
    # These are JD's own words — presence is a red flag.
    "EXPLICIT_NEGATIVE": {
        "phrases": [
            # Title-chaser pattern
            "switching companies every 1.5 years",
            "optimizing for title",
            # Framework enthusiast (no systems thinking)
            "langchain tutorials",
            "how i used langchain",
            "langchain",       # in skills-only context = wrapper flag
            "llamaindex",      # same
            "flowise",
            # Pure consulting
            "tcs", "infosys", "wipro", "accenture", "cognizant",
            "capgemini", "hcl", "tech mahindra",
            # Wrong domain without NLP/IR
            "computer vision without nlp",
            "speech without nlp",
            "primary expertise is computer vision",
            "primary expertise is speech",
            # Closed-source isolation
            "entirely on closed-source",
            "no external validation",
        ],
        "polarity": "negative",
        "weight":   -1.00,
    },
}

# ─────────────────────────────────────────────────────────────────
# TITLE RELEVANCE MAP
# Derived from JD and career context.
# Score 0–5: 5 = perfect match, 0 = irrelevant
# ─────────────────────────────────────────────────────────────────
TITLE_RELEVANCE = {
    # Score 5 — perfect
    "Senior Machine Learning Engineer": 5,
    "Machine Learning Engineer":        5,
    "ML Engineer":                      5,
    "AI Engineer":                      5,
    "Recommendation Systems Engineer":  5,
    "NLP Engineer":                     5,
    "Search Engineer":                  5,
    "Information Retrieval Engineer":   5,
    "Research Engineer":                5,
    "Applied Scientist":                5,
    "AI Research Engineer":             5,
    # Score 4 — strong
    "Data Scientist":                   4,
    "Senior Data Scientist":            4,
    "Applied ML Engineer":              4,
    "Senior AI Engineer":               4,
    # Score 3 — moderate
    "Backend Engineer":                 3,
    "Senior Backend Engineer":          3,
    "Software Engineer":                3,
    "Senior Software Engineer":         3,
    "Data Engineer":                    3,
    "Senior Data Engineer":             3,
    "Analytics Engineer":               3,
    "Full Stack Developer":             3,
    # Score 2 — weak
    "Cloud Engineer":                   2,
    "DevOps Engineer":                  2,
    "MLOps Engineer":                   2,
    "Platform Engineer":                2,
    "Data Analyst":                     2,
    # Score 1 — very weak
    "QA Engineer":                      1,
    "Frontend Engineer":                1,
    "Mobile Developer":                 1,
    "Java Developer":                   1,
    ".NET Developer":                   1,
}

# Titles that are hard disqualifiers regardless of skills
DISQUALIFIED_TITLES = {
    "HR Manager", "HR Executive", "Human Resources Manager",
    "Accountant", "Finance Manager", "CFO",
    "Marketing Manager", "Content Writer", "SEO Specialist",
    "Operations Manager", "Business Analyst",
    "Mechanical Engineer", "Civil Engineer",
    "Graphic Designer", "UI/UX Designer",
    "Sales Executive", "Sales Manager",
    "Customer Support", "Customer Service",
    "Project Manager", "Scrum Master",
    "Teacher", "Professor", "Lecturer",
}

# ─────────────────────────────────────────────────────────────────
# SENIORITY LADDER — for title-chaser detection
# Index = seniority level (higher = more senior)
# ─────────────────────────────────────────────────────────────────
SENIORITY_LADDER = [
    "intern", "junior", "associate", "mid",
    "senior", "staff", "principal", "lead",
    "architect", "director", "vp", "head", "chief",
]

# ─────────────────────────────────────────────────────────────────
# PRODUCT INDUSTRIES  (positive career context signal)
# ─────────────────────────────────────────────────────────────────
PRODUCT_INDUSTRIES = {
    "Software", "Fintech", "E-commerce", "AI/ML", "SaaS",
    "EdTech", "Food Delivery", "HealthTech", "Gaming",
    "Media", "Telecom", "AdTech", "Insurance Tech",
    "Conversational AI", "AI Services", "HealthTech AI",
}

# ─────────────────────────────────────────────────────────────────
# CV / SPEECH / ROBOTICS SKILLS
# Presence as primary domain = JD disqualifier
# ─────────────────────────────────────────────────────────────────
CV_SPEECH_SKILLS = {
    "Image Classification", "Object Detection", "Speech Recognition",
    "TTS", "GANs", "Stable Diffusion", "OpenCV", "YOLO",
    "Robotics", "Computer Vision", "Pose Estimation",
    "Optical Character Recognition", "Face Recognition",
}
