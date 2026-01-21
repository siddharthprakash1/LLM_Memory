"""
Benchmark Scenarios for LLM Memory Systems

Based on industry-standard benchmarks:
- GoodAI LTM Benchmark: Episodic, spatial, personalization
- LoCoMo / LongMemEval-S: Temporal reasoning, preference consistency
- MemoryAgentBench: 4 core competencies
- Mem0 Benchmark: Single-hop, multi-hop, temporal, open-domain
"""

from __future__ import annotations

import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator
from datetime import datetime, timedelta


@dataclass
class BenchmarkSample:
    """A single benchmark sample/question."""
    
    id: str
    scenario_type: str
    context: list[dict[str, str]]  # Conversation history to inject
    query: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    tags: list[str] = field(default_factory=list)


class BaseScenario(ABC):
    """Base class for benchmark scenarios."""
    
    name: str = "base"
    description: str = "Base scenario"
    
    @abstractmethod
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        """Generate benchmark samples."""
        pass
    
    def get_info(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
        }


class SingleHopScenario(BaseScenario):
    """
    Single-hop factual retrieval.
    Tests: Can the system retrieve a directly stated fact?
    
    Based on Mem0 Benchmark single-hop tasks.
    """
    
    name = "single_hop"
    description = "Single-hop factual retrieval - retrieve directly stated facts"
    
    # Fact templates for generation
    FACT_TEMPLATES = [
        {"context": "My name is {name}.", "query": "What is my name?", "answer": "{name}"},
        {"context": "I work at {company}.", "query": "Where do I work?", "answer": "{company}"},
        {"context": "I live in {city}.", "query": "Where do I live?", "answer": "{city}"},
        {"context": "My favorite color is {color}.", "query": "What is my favorite color?", "answer": "{color}"},
        {"context": "I am {age} years old.", "query": "How old am I?", "answer": "{age}"},
        {"context": "My email is {email}.", "query": "What is my email address?", "answer": "{email}"},
        {"context": "I have a {pet} named {pet_name}.", "query": "What is my pet's name?", "answer": "{pet_name}"},
        {"context": "I graduated from {university}.", "query": "Where did I graduate from?", "answer": "{university}"},
        {"context": "My phone number is {phone}.", "query": "What is my phone number?", "answer": "{phone}"},
        {"context": "I drive a {car_color} {car_brand}.", "query": "What car do I drive?", "answer": "{car_color} {car_brand}"},
    ]
    
    # Data pools for generation
    NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    COMPANIES = ["Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla", "OpenAI", "Anthropic", "Couchbase"]
    CITIES = ["New York", "San Francisco", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Singapore", "Dubai"]
    COLORS = ["blue", "red", "green", "purple", "orange", "black", "white", "yellow", "pink", "teal"]
    AGES = ["25", "30", "35", "28", "42", "33", "27", "38", "45", "29"]
    PETS = ["dog", "cat", "parrot", "hamster", "rabbit"]
    PET_NAMES = ["Max", "Luna", "Charlie", "Bella", "Rocky", "Daisy", "Buddy", "Coco"]
    UNIVERSITIES = ["MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "Berkeley", "Yale", "Princeton"]
    CAR_BRANDS = ["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet"]
    CAR_COLORS = ["black", "white", "silver", "blue", "red"]
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        for i in range(count):
            template = random.choice(self.FACT_TEMPLATES)
            
            # Generate values
            values = {
                "name": random.choice(self.NAMES),
                "company": random.choice(self.COMPANIES),
                "city": random.choice(self.CITIES),
                "color": random.choice(self.COLORS),
                "age": random.choice(self.AGES),
                "email": f"{random.choice(self.NAMES).lower()}@example.com",
                "pet": random.choice(self.PETS),
                "pet_name": random.choice(self.PET_NAMES),
                "university": random.choice(self.UNIVERSITIES),
                "phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "car_brand": random.choice(self.CAR_BRANDS),
                "car_color": random.choice(self.CAR_COLORS),
            }
            
            context_text = template["context"].format(**values)
            query = template["query"]
            answer = template["answer"].format(**values)
            
            samples.append(BenchmarkSample(
                id=f"single_hop_{i:04d}",
                scenario_type=self.name,
                context=[{"role": "user", "content": context_text}],
                query=query,
                expected_answer=answer,
                difficulty="easy",
                tags=["factual", "single-hop", "retrieval"],
            ))
        
        return samples


class MultiHopScenario(BaseScenario):
    """
    Multi-hop reasoning requiring inference across multiple memories.
    Tests: Can the system combine information from multiple sources?
    
    Based on Mem0 Benchmark multi-hop tasks.
    """
    
    name = "multi_hop"
    description = "Multi-hop reasoning - combine information across multiple memories"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Multi-hop templates
        templates = [
            {
                "contexts": [
                    "My friend {friend} works at {company}.",
                    "{company} is headquartered in {city}.",
                ],
                "query": "In which city does my friend {friend} work?",
                "answer": "{city}",
            },
            {
                "contexts": [
                    "I bought a {item} from {store}.",
                    "{store} is owned by {owner}.",
                ],
                "query": "Who owns the store where I bought my {item}?",
                "answer": "{owner}",
            },
            {
                "contexts": [
                    "My sister {sister} lives in {city}.",
                    "{city} is famous for its {landmark}.",
                ],
                "query": "What landmark is near where my sister {sister} lives?",
                "answer": "{landmark}",
            },
            {
                "contexts": [
                    "I'm learning {language} from {teacher}.",
                    "{teacher} is from {country}.",
                    "{country}'s capital is {capital}.",
                ],
                "query": "What is the capital of the country where my {language} teacher is from?",
                "answer": "{capital}",
            },
        ]
        
        friends = ["Sarah", "Mike", "Emma", "James", "Olivia"]
        companies = ["TechCorp", "DataInc", "CloudSoft", "AILabs", "DevOps Co"]
        cities = ["Seattle", "Austin", "Boston", "Denver", "Chicago"]
        items = ["laptop", "phone", "camera", "watch", "headphones"]
        stores = ["TechMart", "ElectroShop", "GadgetWorld", "DigitalStore"]
        owners = ["Jeff", "Tim", "Elon", "Mark", "Sundar"]
        sisters = ["Emily", "Sophie", "Anna", "Lucy", "Mia"]
        landmarks = ["Space Needle", "Capitol Building", "Freedom Trail", "Red Rocks", "Bean"]
        languages = ["Spanish", "French", "Japanese", "German", "Mandarin"]
        teachers = ["Mr. Garcia", "Mme. Dupont", "Tanaka-sensei", "Herr Schmidt", "Li Laoshi"]
        countries = ["Spain", "France", "Japan", "Germany", "China"]
        capitals = ["Madrid", "Paris", "Tokyo", "Berlin", "Beijing"]
        
        for i in range(count):
            template = random.choice(templates)
            
            values = {
                "friend": random.choice(friends),
                "company": random.choice(companies),
                "city": random.choice(cities),
                "item": random.choice(items),
                "store": random.choice(stores),
                "owner": random.choice(owners),
                "sister": random.choice(sisters),
                "landmark": random.choice(landmarks),
                "language": random.choice(languages),
                "teacher": random.choice(teachers),
                "country": random.choice(countries),
                "capital": random.choice(capitals),
            }
            
            contexts = [
                {"role": "user", "content": c.format(**values)}
                for c in template["contexts"]
            ]
            
            hops = len(template["contexts"])
            
            samples.append(BenchmarkSample(
                id=f"multi_hop_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=template["query"].format(**values),
                expected_answer=template["answer"].format(**values),
                difficulty="hard" if hops > 2 else "medium",
                tags=["reasoning", "multi-hop", f"{hops}-hop"],
                metadata={"hops": hops},
            ))
        
        return samples


class TemporalScenario(BaseScenario):
    """
    Temporal reasoning and ordering tasks.
    Tests: Can the system handle time-based queries and updates?
    
    Based on LoCoMo temporal tasks and Mem0 temporal reasoning.
    """
    
    name = "temporal"
    description = "Temporal reasoning - handle time-based queries and memory updates"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Temporal templates
        templates = [
            # Recency test
            {
                "type": "recency",
                "contexts": [
                    {"time": -7, "content": "I'm currently reading {book1}."},
                    {"time": -1, "content": "I just started reading {book2}."},
                ],
                "query": "What book am I currently reading?",
                "answer": "{book2}",  # Most recent
            },
            # Update/change test
            {
                "type": "update",
                "contexts": [
                    {"time": -30, "content": "I work at {company1}."},
                    {"time": -2, "content": "I just got a new job at {company2}."},
                ],
                "query": "Where do I currently work?",
                "answer": "{company2}",
            },
            # Historical query
            {
                "type": "historical",
                "contexts": [
                    {"time": -90, "content": "I lived in {city1} for 2 years."},
                    {"time": -30, "content": "I moved to {city2}."},
                ],
                "query": "Where did I live before moving to {city2}?",
                "answer": "{city1}",
            },
            # Sequence ordering
            {
                "type": "sequence",
                "contexts": [
                    {"time": -60, "content": "I visited {place1}."},
                    {"time": -30, "content": "I traveled to {place2}."},
                    {"time": -7, "content": "I went to {place3}."},
                ],
                "query": "What place did I visit after {place1} but before {place3}?",
                "answer": "{place2}",
            },
        ]
        
        books = ["1984", "Dune", "Foundation", "Neuromancer", "Snow Crash", "The Martian"]
        companies = ["Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle"]
        places = ["Paris", "London", "Tokyo", "Rome", "Barcelona", "Sydney", "Dubai", "Singapore"]
        
        for i in range(count):
            template = random.choice(templates)
            
            # Shuffle to get random unique values
            random.shuffle(books)
            random.shuffle(companies)
            random.shuffle(cities)
            random.shuffle(places)
            
            values = {
                "book1": books[0],
                "book2": books[1],
                "company1": companies[0],
                "company2": companies[1],
                "city1": cities[0],
                "city2": cities[1],
                "place1": places[0],
                "place2": places[1],
                "place3": places[2],
            }
            
            base_time = datetime.now()
            contexts = []
            for ctx in template["contexts"]:
                timestamp = base_time + timedelta(days=ctx["time"])
                contexts.append({
                    "role": "user",
                    "content": ctx["content"].format(**values),
                    "timestamp": timestamp.isoformat(),
                })
            
            samples.append(BenchmarkSample(
                id=f"temporal_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=template["query"].format(**values),
                expected_answer=template["answer"].format(**values),
                difficulty="medium",
                tags=["temporal", template["type"], "time-aware"],
                metadata={"temporal_type": template["type"]},
            ))
        
        return samples


class ConflictScenario(BaseScenario):
    """
    Conflict detection and resolution tasks.
    Tests: Can the system identify and resolve contradictory information?
    
    Based on MemoryAgentBench conflict resolution competency.
    """
    
    name = "conflict"
    description = "Conflict resolution - detect and resolve contradictory information"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Conflict templates
        templates = [
            # Direct contradiction
            {
                "type": "direct_contradiction",
                "contexts": [
                    {"content": "I am {age1} years old.", "time": -30},
                    {"content": "I am {age2} years old.", "time": -1},
                ],
                "query": "How old am I?",
                "answer": "{age2}",  # More recent wins
                "resolution": "recency",
            },
            # Preference change
            {
                "type": "preference_change",
                "contexts": [
                    {"content": "I hate {food}. It's disgusting.", "time": -60},
                    {"content": "I've started to really enjoy {food} lately.", "time": -5},
                ],
                "query": "Do I like {food}?",
                "answer": "yes",
                "resolution": "recency",
            },
            # Factual update
            {
                "type": "factual_update",
                "contexts": [
                    {"content": "My phone number is {phone1}.", "time": -90},
                    {"content": "I changed my number. It's now {phone2}.", "time": -7},
                ],
                "query": "What is my current phone number?",
                "answer": "{phone2}",
                "resolution": "explicit_update",
            },
            # Ambiguous conflict (should ask for clarification or use confidence)
            {
                "type": "ambiguous",
                "contexts": [
                    {"content": "I think my favorite movie is {movie1}.", "time": -20},
                    {"content": "Actually, {movie2} might be my favorite.", "time": -10},
                ],
                "query": "What is my favorite movie?",
                "answer": "{movie2}",  # More recent, but hedged
                "resolution": "confidence",
            },
        ]
        
        ages = ["25", "26", "30", "31", "35", "36", "40", "41"]
        foods = ["sushi", "pizza", "curry", "tacos", "pasta", "steak"]
        phones = [f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}" for _ in range(10)]
        movies = ["Inception", "The Matrix", "Interstellar", "Pulp Fiction", "The Godfather", "Fight Club"]
        
        for i in range(count):
            template = random.choice(templates)
            
            age_pair = random.sample(ages, 2)
            phone_pair = random.sample(phones, 2)
            movie_pair = random.sample(movies, 2)
            
            values = {
                "age1": age_pair[0],
                "age2": age_pair[1],
                "food": random.choice(foods),
                "phone1": phone_pair[0],
                "phone2": phone_pair[1],
                "movie1": movie_pair[0],
                "movie2": movie_pair[1],
            }
            
            base_time = datetime.now()
            contexts = []
            for ctx in template["contexts"]:
                timestamp = base_time + timedelta(days=ctx["time"])
                contexts.append({
                    "role": "user",
                    "content": ctx["content"].format(**values),
                    "timestamp": timestamp.isoformat(),
                })
            
            samples.append(BenchmarkSample(
                id=f"conflict_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=template["query"].format(**values),
                expected_answer=template["answer"].format(**values),
                difficulty="hard",
                tags=["conflict", template["type"], template["resolution"]],
                metadata={
                    "conflict_type": template["type"],
                    "expected_resolution": template["resolution"],
                },
            ))
        
        return samples


class PreferenceScenario(BaseScenario):
    """
    Preference consistency and personalization tasks.
    Tests: Can the system maintain consistent user preferences?
    
    Based on GoodAI LTM personalization and LoCoMo preference tasks.
    """
    
    name = "preference"
    description = "Preference consistency - maintain and apply user preferences"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Preference templates
        templates = [
            # Direct preference recall
            {
                "contexts": ["I prefer {pref_a} over {pref_b}."],
                "query": "If I had to choose between {pref_a} and {pref_b}, which would I pick?",
                "answer": "{pref_a}",
            },
            # Implicit preference from context
            {
                "contexts": [
                    "I always order {drink} at coffee shops.",
                    "I've been to Starbucks 50 times this year.",
                ],
                "query": "What should I order at a new coffee shop?",
                "answer": "{drink}",
            },
            # Preference with constraints
            {
                "contexts": [
                    "I'm vegetarian.",
                    "My favorite cuisine is {cuisine}.",
                ],
                "query": "Can you recommend a restaurant type for me?",
                "answer": "vegetarian {cuisine}",
            },
            # Negative preference
            {
                "contexts": ["I really dislike {dislike}. Never recommend it to me."],
                "query": "Should I try {dislike}?",
                "answer": "no",
            },
        ]
        
        preferences = [
            ("coffee", "tea"), ("morning", "evening"), ("beach", "mountains"),
            ("cats", "dogs"), ("reading", "watching TV"), ("summer", "winter"),
        ]
        drinks = ["latte", "cappuccino", "americano", "espresso", "cold brew"]
        cuisines = ["Italian", "Japanese", "Mexican", "Indian", "Thai", "Chinese"]
        dislikes = ["horror movies", "spicy food", "loud music", "crowded places", "early mornings"]
        
        for i in range(count):
            template = random.choice(templates)
            pref = random.choice(preferences)
            
            values = {
                "pref_a": pref[0],
                "pref_b": pref[1],
                "drink": random.choice(drinks),
                "cuisine": random.choice(cuisines),
                "dislike": random.choice(dislikes),
            }
            
            contexts = [
                {"role": "user", "content": c.format(**values)}
                for c in template["contexts"]
            ]
            
            samples.append(BenchmarkSample(
                id=f"preference_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=template["query"].format(**values),
                expected_answer=template["answer"].format(**values),
                difficulty="medium",
                tags=["preference", "personalization", "consistency"],
            ))
        
        return samples


class EpisodicScenario(BaseScenario):
    """
    Episodic memory tasks - recall of specific events and experiences.
    Tests: Can the system remember and retrieve specific episodes?
    
    Based on GoodAI LTM episodic memory tasks.
    """
    
    name = "episodic"
    description = "Episodic memory - recall specific events and experiences"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Episode templates
        templates = [
            # Event recall
            {
                "contexts": [
                    "Yesterday I went to {place} with {person}.",
                    "We had a great time and ate {food}.",
                ],
                "query": "Who did I go to {place} with?",
                "answer": "{person}",
            },
            # Temporal event query
            {
                "contexts": [
                    "Last week I attended a {event_type} at {venue}.",
                    "The main speaker was {speaker}.",
                ],
                "query": "What event did I attend last week?",
                "answer": "{event_type} at {venue}",
            },
            # Emotional context
            {
                "contexts": [
                    "I had an amazing experience at {restaurant}.",
                    "The {dish} was incredible - best I've ever had!",
                ],
                "query": "What made my experience at {restaurant} special?",
                "answer": "{dish}",
            },
            # Multi-part episode
            {
                "contexts": [
                    "My trip to {destination} was unforgettable.",
                    "I stayed at {hotel} for {nights} nights.",
                    "The highlight was visiting {attraction}.",
                ],
                "query": "What was the highlight of my {destination} trip?",
                "answer": "{attraction}",
            },
        ]
        
        places = ["the park", "the beach", "the museum", "downtown", "the mall"]
        people = ["Sarah", "Mike", "my brother", "my mom", "my best friend"]
        foods = ["pizza", "sushi", "burgers", "tacos", "pasta"]
        event_types = ["conference", "concert", "workshop", "meetup", "lecture"]
        venues = ["the convention center", "Madison Square Garden", "the community center", "the university"]
        speakers = ["Dr. Smith", "the CEO", "a famous author", "an industry expert"]
        restaurants = ["The Italian Place", "Sushi Master", "The Steakhouse", "Cafe Luna"]
        dishes = ["truffle pasta", "wagyu steak", "omakase sushi", "lobster bisque"]
        destinations = ["Paris", "Tokyo", "New York", "Bali", "Iceland"]
        hotels = ["The Grand Hotel", "Marriott", "a charming Airbnb", "the Hilton"]
        nights_options = ["3", "5", "7", "10"]
        attractions = ["the Eiffel Tower", "Mount Fuji", "Central Park", "the Northern Lights"]
        
        for i in range(count):
            template = random.choice(templates)
            
            values = {
                "place": random.choice(places),
                "person": random.choice(people),
                "food": random.choice(foods),
                "event_type": random.choice(event_types),
                "venue": random.choice(venues),
                "speaker": random.choice(speakers),
                "restaurant": random.choice(restaurants),
                "dish": random.choice(dishes),
                "destination": random.choice(destinations),
                "hotel": random.choice(hotels),
                "nights": random.choice(nights_options),
                "attraction": random.choice(attractions),
            }
            
            contexts = [
                {"role": "user", "content": c.format(**values)}
                for c in template["contexts"]
            ]
            
            samples.append(BenchmarkSample(
                id=f"episodic_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=template["query"].format(**values),
                expected_answer=template["answer"].format(**values),
                difficulty="medium",
                tags=["episodic", "event-recall", "experience"],
            ))
        
        return samples


class LongRangeScenario(BaseScenario):
    """
    Long-range dependency tasks across many sessions/turns.
    Tests: Can the system maintain coherence over extended interactions?
    
    Based on MemoryAgentBench long-range understanding.
    """
    
    name = "long_range"
    description = "Long-range understanding - maintain coherence across many sessions"
    
    def generate_samples(self, count: int, seed: int | None = None) -> list[BenchmarkSample]:
        if seed is not None:
            random.seed(seed)
        
        samples = []
        
        # Generate samples with increasing context lengths
        topics = ["project", "learning", "relationship", "goal", "hobby"]
        
        for i in range(count):
            topic = random.choice(topics)
            num_contexts = random.randint(5, 15)  # Many context items
            
            if topic == "project":
                project_name = f"Project {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}"
                contexts = [
                    {"role": "user", "content": f"I started working on {project_name}."},
                ]
                milestones = ["planning", "design", "implementation", "testing", "deployment"]
                for j, milestone in enumerate(milestones[:num_contexts-1]):
                    contexts.append({
                        "role": "user",
                        "content": f"Completed the {milestone} phase of {project_name}.",
                    })
                
                query = f"What phases have I completed on {project_name}?"
                answer = ", ".join(milestones[:num_contexts-1])
                
            elif topic == "learning":
                skill = random.choice(["Python", "piano", "Spanish", "cooking", "chess"])
                contexts = [{"role": "user", "content": f"I started learning {skill}."}]
                levels = ["basics", "intermediate concepts", "advanced techniques", "expert level", "mastery"]
                for level in levels[:num_contexts-1]:
                    contexts.append({
                        "role": "user",
                        "content": f"I've progressed to {level} in {skill}.",
                    })
                
                query = f"What level am I at in {skill}?"
                answer = levels[min(num_contexts-2, len(levels)-1)]
                
            else:
                # Generic long-range
                contexts = []
                facts = []
                for j in range(num_contexts):
                    fact = f"fact_{j}_{random.randint(100, 999)}"
                    facts.append(fact)
                    contexts.append({
                        "role": "user",
                        "content": f"Remember this: {fact}",
                    })
                
                target_idx = random.randint(0, len(facts)-1)
                query = f"What was fact number {target_idx + 1} that I asked you to remember?"
                answer = facts[target_idx]
            
            samples.append(BenchmarkSample(
                id=f"long_range_{i:04d}",
                scenario_type=self.name,
                context=contexts,
                query=query,
                expected_answer=answer,
                difficulty="hard",
                tags=["long-range", "multi-session", "coherence"],
                metadata={"context_length": len(contexts)},
            ))
        
        return samples


# Scenario registry
SCENARIOS = {
    "single_hop": SingleHopScenario(),
    "multi_hop": MultiHopScenario(),
    "temporal": TemporalScenario(),
    "conflict": ConflictScenario(),
    "preference": PreferenceScenario(),
    "episodic": EpisodicScenario(),
    "long_range": LongRangeScenario(),
}


def get_scenario(name: str) -> BaseScenario:
    """Get a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> list[dict]:
    """List all available scenarios."""
    return [s.get_info() for s in SCENARIOS.values()]
