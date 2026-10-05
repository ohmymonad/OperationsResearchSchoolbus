# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Operations Research (OR) course project implementing a solution to **Problem 10: ScuolaBus I** — a school bus stop assignment and routing optimization problem.

**Problem**: Assign children to bus stops (within distance ε of their home) and find the minimum-length bus route visiting all stops with assigned children, ending at school.

**Constraint**: Each child must have at least 3 viable stops within radius ε of their residence.

## Requirements from Course Specification

### Core Deliverables
1. **Abstract problem formulation** using mathematical notation:
   - Decision variables (which stop each child is assigned to)
   - Objective function (minimize route length)
   - Constraints (distance, stop coverage, routing)

2. **Algorithmic approach**: Implement at least one resolution method from course material (heuristics, metaheuristics, exact methods, etc.)

3. **Implementation**: Apply the chosen approach to solve the problem

4. **Experimental validation**: Test on at least one instance

### Rules & Constraints
- **Libraries**: May use data structure and algorithm libraries; cannot use solver libraries for entire problem (solvers only allowed for estimating error or within math-heuristics)
- **No logic programming**: Implementation must use imperative languages (Python, C++, Java, Go, Rust, etc.) — not Prolog or ASP
- **Grading**: Project counts for max 21 points toward final grade (17 if using alternative compilation mode)

## Recommended Architecture

### Phase 1: Problem Modeling
- Define the mathematical model (decision variables, objective, constraints)
- Choose input/output data format (JSON, CSV, or plain text)

### Phase 2: Core Algorithm Components
- **Instance generator**: Create random problem instances or use existing benchmarks
  - Reference: OR-Library (people.brunel.ac.uk/~mastjjb/jeb/info.html)
  - VRP instances: neo.lcc.uma.es/vrp/vrp-instances/
- **Distance computation**: Euclidean or graph-based distance between children's homes, stops, and school
- **Assignment solver**: Assign each child to a feasible stop
- **Route solver**: Solve TSP/VRP variant to find minimum-length route through assigned stops
- **Evaluator**: Compute objective value for a given solution

### Phase 3: Solution Methods
Common OR approaches for this hybrid problem:
- **Constructive heuristics** (greedy stop assignment → nearest-neighbor routing)
- **Local search** (2-opt on route + neighborhood search on assignment)
- **Metaheuristics** (simulated annealing, genetic algorithms, tabu search)
- **Exact methods** (branch-and-bound, integer programming formulation with solver for comparison)

### Phase 4: Testing & Validation
- Generate multiple problem instances
- Run algorithm on each instance
- Report: solution quality, runtime, comparison to baseline or known bounds
- Document convergence/sensitivity to parameters

## Development Notes

### Data Structure Recommendations
- Graph representation: adjacency list or matrix for distances
- Assignment: array mapping child → stop
- Route: permutation or sequence of stops

### Common Pitfalls
- Forgetting to validate assignment feasibility (distance ε constraint)
- Not checking that each child has ≥3 viable options before assigning
- Confusing stop assignment (combinatorial) with route optimization (TSP-like)
- Using solver libraries for the entire problem instead of just error estimation

### Expected Problem Structure
```
Input:
- Set of children B with residences
- Set of potential stops N with locations
- School location
- Distance threshold ε

Output:
- Assignment of each child to a stop (within ε)
- Bus route through stops + school
- Route length / total cost
```

## Testing & Validation

### Minimal Test Case
Create a small instance (5-10 children, 3-5 stops) where optimal or near-optimal solution is manually verifiable.

### Medium Instance
Use benchmark data or generator to test on 50-100 children with 20-30 stops.

### Reporting
- Show instance parameters (|B|, |N|, ε)
- Report solution: assignment, route order, total distance
- Compare heuristic vs. exact/baseline if applicable
