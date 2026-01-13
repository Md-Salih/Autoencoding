"""
Comprehensive Data Loader for Student Model Training
Combines multiple diverse text sources to improve generalization
"""

import random
from typing import List, Tuple


def get_common_knowledge_sentences() -> List[str]:
    """General knowledge and common facts"""
    return [
        "The sun rises in the east and sets in the west",
        "Water boils at one hundred degrees celsius",
        "The earth revolves around the sun",
        "Gravity pulls objects toward the ground",
        "Plants need sunlight water and nutrients to grow",
        "The human body has two hundred and six bones",
        "The capital of France is Paris",
        "The Pacific Ocean is the largest ocean",
        "Shakespeare wrote Romeo and Juliet",
        "The moon orbits around the earth",
        "Light travels faster than sound",
        "Ice melts when temperature rises above zero",
        "Birds have feathers and hollow bones",
        "The Amazon rainforest produces twenty percent of oxygen",
        "Mount Everest is the tallest mountain",
        "The Great Wall of China is visible from space",
        "Albert Einstein developed the theory of relativity",
        "DNA contains genetic information",
        "The Nile is the longest river",
        "Antarctica is the coldest continent",
    ]


def get_daily_life_sentences() -> List[str]:
    """Everyday activities and common situations"""
    return [
        "She drinks coffee every morning before work",
        "The children play in the park after school",
        "He reads the newspaper during breakfast",
        "They walk their dog around the neighborhood",
        "The teacher explains the lesson to students",
        "Mom cooks dinner for the whole family",
        "Dad drives to work in heavy traffic",
        "The doctor examines patients in the clinic",
        "Students study hard for their final exams",
        "The chef prepares delicious meals in the kitchen",
        "She listens to music while jogging",
        "He watches television before going to bed",
        "They celebrate birthdays with cake and candles",
        "The farmer grows vegetables in the field",
        "The mechanic repairs cars in the garage",
        "She shops for groceries at the supermarket",
        "He exercises at the gym three times weekly",
        "The baby sleeps peacefully in the crib",
        "Students raise their hands to ask questions",
        "The postman delivers mail every afternoon",
    ]


def get_technology_sentences() -> List[str]:
    """Technology and computing related sentences"""
    return [
        "Computers process data using binary code",
        "The internet connects people around the world",
        "Smartphones have powerful cameras and processors",
        "Artificial intelligence learns from data patterns",
        "Social media platforms enable global communication",
        "Cloud computing stores data on remote servers",
        "Robots automate repetitive manufacturing tasks",
        "Electric vehicles reduce carbon emissions significantly",
        "Virtual reality creates immersive digital experiences",
        "Blockchain technology ensures secure transactions",
        "Machine learning algorithms improve with training",
        "Quantum computers solve complex mathematical problems",
        "Solar panels convert sunlight into electricity",
        "Satellites orbit earth for communication purposes",
        "Autonomous vehicles use sensors and cameras",
        "Streaming services deliver content over internet",
        "Touchscreens respond to finger gestures accurately",
        "Wireless networks transmit data without cables",
        "Drones capture aerial photographs and videos",
        "Biometric systems identify users by fingerprints",
    ]


def get_nature_sentences() -> List[str]:
    """Nature, animals, and environment"""
    return [
        "Trees produce oxygen through photosynthesis process",
        "Bees pollinate flowers while collecting nectar",
        "Dolphins are intelligent marine mammals",
        "Eagles have excellent vision for hunting",
        "Elephants remember locations of water sources",
        "Butterflies undergo metamorphosis from caterpillars",
        "Whales migrate thousands of miles annually",
        "Wolves hunt in packs for survival",
        "Coral reefs provide habitat for marine life",
        "Mountains form through tectonic plate movement",
        "Forests absorb carbon dioxide from atmosphere",
        "Rivers flow from mountains to oceans",
        "Deserts receive very little annual rainfall",
        "Seasons change due to earth's tilt",
        "Lightning occurs during thunderstorms frequently",
        "Tsunamis result from underwater earthquakes",
        "Glaciers contain most of earth's fresh water",
        "Volcanoes erupt molten lava and ash",
        "Tornadoes form from severe thunderstorms",
        "Rainbows appear when sunlight refracts through rain",
    ]


def get_education_sentences() -> List[str]:
    """Education and learning related"""
    return [
        "Students learn mathematics using various methods",
        "Reading improves vocabulary and comprehension skills",
        "Science experiments demonstrate fundamental principles",
        "History teaches lessons from past events",
        "Geography studies earth's physical features",
        "Literature explores human emotions and experiences",
        "Art develops creativity and self expression",
        "Music enhances cognitive and emotional development",
        "Physical education promotes health and fitness",
        "Language learning requires consistent practice",
        "Critical thinking helps solve complex problems",
        "Collaboration improves group project outcomes",
        "Research requires gathering and analyzing data",
        "Writing skills improve through regular practice",
        "Public speaking builds confidence and communication",
        "Time management helps students meet deadlines",
        "Note taking improves information retention",
        "Study groups enhance learning through discussion",
        "Online courses provide flexible learning options",
        "Exams assess student knowledge and understanding",
    ]


def get_health_sentences() -> List[str]:
    """Health and wellness"""
    return [
        "Regular exercise strengthens muscles and bones",
        "Healthy diet includes fruits and vegetables",
        "Sleep is essential for mental health",
        "Drinking water keeps body hydrated properly",
        "Meditation reduces stress and anxiety",
        "Vitamins support immune system function",
        "Yoga improves flexibility and balance",
        "Proper posture prevents back pain",
        "Handwashing prevents spread of germs",
        "Vaccinations protect against serious diseases",
        "Dental care prevents cavities and decay",
        "Sunscreen protects skin from harmful rays",
        "Fresh air improves respiratory health",
        "Regular checkups detect health problems early",
        "Antibiotics fight bacterial infections effectively",
        "Protein builds and repairs body tissues",
        "Cardio exercise strengthens the heart",
        "Mental health affects overall wellbeing",
        "Balanced lifestyle includes work and rest",
        "Nutrition labels help make healthy choices",
    ]


def get_business_sentences() -> List[str]:
    """Business and economics"""
    return [
        "Companies develop products to meet customer needs",
        "Marketing promotes products through various channels",
        "Employees receive salaries for their work",
        "Investors buy stocks to earn returns",
        "Banks provide loans to businesses and individuals",
        "Supply and demand determine market prices",
        "Entrepreneurs start new businesses with innovation",
        "Competition drives companies to improve quality",
        "Exports generate revenue from foreign markets",
        "Inflation increases prices over time gradually",
        "Budget planning helps manage financial resources",
        "Contracts establish legal agreements between parties",
        "Insurance protects against unexpected losses",
        "Retail stores sell products directly to consumers",
        "Manufacturing converts raw materials into products",
        "E-commerce enables online shopping and payments",
        "Advertising influences consumer purchasing decisions",
        "Quality control ensures product standards",
        "Customer service builds brand loyalty",
        "Profit margins indicate business performance",
    ]


def get_travel_sentences() -> List[str]:
    """Travel and geography"""
    return [
        "Airplanes transport passengers across continents quickly",
        "Hotels provide accommodation for travelers",
        "Tourists visit famous landmarks and monuments",
        "Passports are required for international travel",
        "Maps help navigate unfamiliar locations",
        "Beaches attract visitors seeking relaxation",
        "Museums display historical artifacts and art",
        "Restaurants serve local cuisine and specialties",
        "Trains connect cities across vast distances",
        "Airports handle millions of passengers daily",
        "Cruise ships offer luxury ocean voyages",
        "National parks preserve natural landscapes",
        "Cities feature diverse cultures and lifestyles",
        "Mountains provide opportunities for hiking",
        "Islands offer tropical vacation experiences",
        "Festivals celebrate cultural traditions worldwide",
        "Language barriers challenge international communication",
        "Currency exchange enables foreign transactions",
        "Souvenirs remind travelers of their journeys",
        "Travel insurance covers unexpected emergencies",
    ]


def get_sports_sentences() -> List[str]:
    """Sports and athletics"""
    return [
        "Football teams compete for championship titles",
        "Basketball players dribble and shoot hoops",
        "Tennis requires agility and quick reflexes",
        "Swimming builds endurance and muscle strength",
        "Athletes train rigorously for competitions",
        "Coaches develop strategies and game plans",
        "Referees enforce rules during matches",
        "Fans cheer loudly for their teams",
        "Stadiums accommodate thousands of spectators",
        "Olympics unite athletes from all nations",
        "Marathons test runners physical and mental stamina",
        "Cycling races cover long challenging distances",
        "Boxing matches require strength and technique",
        "Golf demands precision and concentration",
        "Baseball involves batting pitching and fielding",
        "Cricket matches can last several days",
        "Hockey players skate at high speeds",
        "Gymnastics showcases flexibility and grace",
        "Wrestling combines power and strategy",
        "Martial arts teach discipline and self defense",
    ]


def get_all_comprehensive_sentences() -> List[str]:
    """Combine all sentence sources"""
    all_sentences = []
    all_sentences.extend(get_common_knowledge_sentences())
    all_sentences.extend(get_daily_life_sentences())
    all_sentences.extend(get_technology_sentences())
    all_sentences.extend(get_nature_sentences())
    all_sentences.extend(get_education_sentences())
    all_sentences.extend(get_health_sentences())
    all_sentences.extend(get_business_sentences())
    all_sentences.extend(get_travel_sentences())
    all_sentences.extend(get_sports_sentences())
    
    # Add Wikipedia sentences if available
    try:
        from data_loader import get_wiki_sentences
        wiki_sentences = get_wiki_sentences(50)  # Get 50 Wikipedia sentences
        all_sentences.extend(wiki_sentences)
    except:
        pass
    
    return all_sentences


def create_mlm_pairs(sentences: List[str], mask_prob: float = 0.15) -> List[Tuple[str, str]]:
    """Create masked language modeling pairs from sentences"""
    pairs = []
    
    for sent in sentences:
        words = sent.split()
        if len(words) < 3:
            continue
        
        # Mask each position once
        for i in range(len(words)):
            masked = words.copy()
            masked[i] = "[MASK]"
            masked_sent = " ".join(masked)
            pairs.append((masked_sent, sent))
    
    return pairs


def get_comprehensive_mlm_dataset() -> List[Tuple[str, str]]:
    """Get comprehensive MLM dataset for training"""
    sentences = get_all_comprehensive_sentences()
    pairs = create_mlm_pairs(sentences)
    
    print(f"Generated {len(pairs)} training examples from {len(sentences)} sentences")
    print(f"Dataset covers: general knowledge, daily life, technology, nature,")
    print(f"education, health, business, travel, sports, and more")
    
    return pairs


if __name__ == "__main__":
    # Test the data loader
    pairs = get_comprehensive_mlm_dataset()
    print(f"\nTotal training pairs: {len(pairs)}")
    print("\nSample pairs:")
    for i in range(min(5, len(pairs))):
        masked, original = pairs[i]
        print(f"{i+1}. Masked: {masked}")
        print(f"   Target: {original}\n")
