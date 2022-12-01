class BoidsColonyState:
    def __init__(self, positions, headings, velocities, is_leader) -> None:
        self.positions = positions
        self.headings = headings
        self.velocities = velocities
        self.is_leader = is_leader


class StateBounds:
    def __init__(self, map_dimensions, min_velocity, max_velocity, max_acceleration, max_angular_velocity,
                 num_leaders, num_followers) -> None:
        self.map_dimensions = map_dimensions
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        self.num_total = num_leaders + num_followers
