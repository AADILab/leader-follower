from enum import IntEnum
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from leader_follower.agents.colony_helpers import BoidsColonyState, StateBounds


class BoidSpawnRule(IntEnum):
    Individual = 0  # Defer to individual rules for position, velocity, heading
    UniformRandom = 1  # Use uniform random for all state variables
    Circle = 2  # Positions in a circle with fixed velocity and random headings
    Set = 3  # Preset specific values for all state variables


class PositionRule(IntEnum):
    UniformRandom = 0
    CenterCircle = 1
    Set = 2


class VelocityRule(IntEnum):
    UniformRandom = 0
    FixedStart = 1
    Set = 2


class HeadingRule(IntEnum):
    UniformRandom = 0
    Set = 1
    FixedStart = 2


class BoidSpawner:
    def __init__(self,
                 # Boundaries for any generated state
                 bounds: StateBounds,
                 # Rules for spawning boids
                 spawn_rule: Union[BoidSpawnRule, str],
                 position_rule: Optional[Union[PositionRule, str]] = None,
                 velocity_rule: Optional[Union[VelocityRule, str]] = None,
                 heading_rule: Optional[Union[HeadingRule, str]] = None,
                 # Whether to fix the spawn
                 fix_all: bool = False,
                 fix_positions: bool = False,
                 fix_velocities: bool = False,
                 fix_headings: bool = False,
                 # Position parameters
                 radius_fraction: Optional[float] = None,
                 leader_positions: Optional[List[List[float]]] = None,
                 follower_positions: Optional[List[List[float]]] = None,
                 # Velocity parameters
                 velocity_fraction: Optional[float] = None,
                 leader_velocities: Optional[List[float]] = None,
                 follower_velocities: Optional[List[float]] = None,
                 # Heading parameters
                 leader_headings: Optional[List[float]] = None,
                 follower_headings: Optional[List[float]] = None,
                 start_heading: Optional[float] = None
                 ) -> None:

        if type(spawn_rule) == str:
            spawn_rule = BoidSpawnRule[spawn_rule]
        if type(position_rule) == str:
            position_rule = PositionRule[position_rule]
        if type(velocity_rule) == str:
            velocity_rule = VelocityRule[velocity_rule]
        if type(heading_rule) == str:
            heading_rule = HeadingRule[heading_rule]

        self.bounds = bounds
        self.spawn_rule = spawn_rule

        self.radius_fraction = radius_fraction
        if leader_positions is not None:
            self.leader_positions = np.array(leader_positions, dtype=float)
        else:
            self.leader_positions = None
        if follower_positions is not None:
            self.follower_positions = np.array(follower_positions, dtype=float)
        else:
            self.follower_positions = None

        self.velocity_fraction = velocity_fraction
        self.leader_velocities = leader_velocities
        self.follower_velocities = follower_velocities

        self.leader_headings = leader_headings
        self.follower_headings = follower_headings
        self.start_heading = start_heading

        # Set rules for position, velocity, heading using overall spawn rule
        if self.spawn_rule.value == BoidSpawnRule.Individual.value:
            self.position_rule = position_rule
            self.velocity_rule = velocity_rule
            self.heading_rule = heading_rule

        elif self.spawn_rule.value == BoidSpawnRule.UniformRandom.value:
            self.position_rule = PositionRule.UniformRandom
            self.velocity_rule = VelocityRule.UniformRandom
            self.heading_rule = HeadingRule.UniformRandom

        elif self.spawn_rule.value == BoidSpawnRule.Circle.value:
            self.position_rule = PositionRule.CenterCircle
            self.velocity_rule = VelocityRule.FixedStart
            self.heading_rule = HeadingRule.UniformRandom

        elif self.spawn_rule.value == BoidSpawnRule.Set.value:
            self.position_rule = PositionRule.Set
            self.velocity_rule = VelocityRule.Set
            self.heading_rule = HeadingRule.Set

        self.fix_all = fix_all
        if self.fix_all:
            fix_positions = True
            fix_velocities = True
            fix_headings = True

        self.fix_positions = fix_positions
        self.fix_velocities = fix_velocities
        self.fix_headings = fix_headings

        # Set fixed start spawns if flag is set
        if self.fix_positions:
            self.fixed_positions = self.gen_positions()

        if self.fix_velocities:
            self.fixed_velocities = self.gen_velocities()

        if self.fix_headings:
            self.fixed_headings = self.gen_headings()

        self.fixed_is_leader = self.gen_leaders()
        return

    def gen_positions(self) -> NDArray[np.float64]:
        if self.position_rule.value == PositionRule.UniformRandom.value:
            return np.hstack((
                np.random.uniform(0, self.bounds.map_dimensions[0], size=(self.bounds.num_total, 1)),
                np.random.uniform(0, self.bounds.map_dimensions[1], size=(self.bounds.num_total, 1))
            ))
        elif self.position_rule.value == PositionRule.CenterCircle.value:
            theta = np.random.uniform(0, 2 * np.pi, size=(self.bounds.num_total, 1))
            radius = self.radius_fraction * np.min(self.bounds.map_dimensions) / 2
            radii = np.random.uniform(0, radius, size=(self.bounds.num_total, 1))
            return self.bounds.map_dimensions / 2 + np.hstack((
                radii * np.cos(theta),
                radii * np.sin(theta)
            ))
        elif self.position_rule.value == PositionRule.Set.value:
            if len(self.follower_positions) == 0:
                return self.leader_positions.copy()
            elif len(self.leader_positions) == 0:
                return self.follower_positions.copy()
            else:
                return np.vstack((self.leader_positions, self.follower_positions)).copy()

    def gen_velocities(self) -> NDArray[np.float64]:
        if self.velocity_rule.value == VelocityRule.UniformRandom.value:
            return np.random.uniform(self.bounds.min_velocity, self.bounds.max_velocity, size=self.bounds.num_total)
        elif self.velocity_rule.value == VelocityRule.FixedStart.value:
            velocity = self.velocity_fraction * (
                        self.bounds.max_velocity - self.bounds.min_velocity) + self.bounds.min_velocity
            return velocity * np.ones(self.bounds.num_total)

    def gen_headings(self):
        if self.heading_rule.value == HeadingRule.UniformRandom.value:
            return np.random.uniform(0, 2 * np.pi, size=self.bounds.num_total)
        elif self.heading_rule.value == HeadingRule.FixedStart.value:
            return self.start_heading * np.ones(self.bounds.num_total)
        elif self.heading_rule.value == HeadingRule.Set.value:
            if self.follower_headings is None:
                return self.leader_headings.copy()
            elif self.leader_headings is None:
                return self.follower_headings.copy()
            else:
                return np.hstack((self.leader_headings, self.follower_headings)).copy()

    def gen_leaders(self) -> NDArray[np.bool_]:
        # Leaders, Followers
        return np.array(
            [True for _ in range(self.bounds.num_leaders)] + [False for _ in range(self.bounds.num_followers)])

    def spawn_state(self):
        if self.fix_positions:
            positions = self.fixed_positions.copy()
        else:
            positions = self.gen_positions()

        if self.fix_velocities:
            velocities = self.fixed_velocities.copy()
        else:
            velocities = self.gen_velocities()

        if self.fix_headings:
            headings = self.fixed_headings.copy()
        else:
            headings = self.gen_headings()
        is_leader = self.fixed_is_leader
        return BoidsColonyState(positions=positions, headings=headings, velocities=velocities, is_leader=is_leader)
