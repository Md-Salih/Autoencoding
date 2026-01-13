# A small, original 10-example dataset for the "Lab" (toy) runs.
#
# The original SAMPLE1–SAMPLE10 from the lab handout are kept in the README
# for reference only; the code trains on these synthetic examples by default.

from __future__ import annotations

from typing import List, Tuple


MLM_PAIRS: List[Tuple[str, str]] = [
    ("The chef adds [MASK] to the soup", "The chef adds salt to the soup"),
    ("The train arrives at [MASK] station", "The train arrives at central station"),
    ("A telescope helps us see [MASK] galaxies", "A telescope helps us see distant galaxies"),
    ("Regular sleep improves [MASK] focus", "Regular sleep improves overall focus"),
    ("The program uses [MASK] variables", "The program uses integer variables"),
    ("Wind turbines generate [MASK] power", "Wind turbines generate clean power"),
    ("The coach plans a [MASK] strategy", "The coach plans a winning strategy"),
    ("Recycling reduces [MASK] waste", "Recycling reduces plastic waste"),
    ("The musician plays a [MASK] melody", "The musician plays a gentle melody"),
    ("The robot sorts [MASK] items", "The robot sorts small items"),
]


CLS_PAIRS: List[Tuple[str, str]] = [
    ("The chef adds [MASK] to the soup", "Food"),
    ("The train arrives at [MASK] station", "Travel"),
    ("A telescope helps us see [MASK] galaxies", "Space"),
    ("Regular sleep improves [MASK] focus", "Health"),
    ("The program uses [MASK] variables", "Computing"),
    ("Wind turbines generate [MASK] power", "Energy"),
    ("The coach plans a [MASK] strategy", "Sports"),
    ("Recycling reduces [MASK] waste", "Environment"),
    ("The musician plays a [MASK] melody", "Music"),
    ("The robot sorts [MASK] items", "Robotics"),
]
