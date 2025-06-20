//! Physics simulation for circles using verlet integration.
//!
//! # Usage
//! ```
//! extern crate verlet;
//! use verlet::*;
//!
//! fn main() {
//!   let mut bodies = vec![];
//!
//!   for i in 0..10 {
//!     bodies.push(Body {
//!       current_position: Vector2(i as f32, 0.0),
//!       radius: 2.0,
//!       payload: (),
//!       ..Default::default()
//!     })
//!   }
//!
//!   // Simulate the bodies 100 times over the course of 30 seconds
//!   simulate(
//!     bodies,
//!     100,
//!     |c| constraint_circle(c, Vector2(0.0, 0.0), 10.0),
//!     Vector2(0.0, 5.0),
//!     std::time::Duration::from_secs(30),
//!   );
//! }
//! ```

/// An (x, y) coordinate
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Vector2(pub f32, pub f32);

impl Default for Vector2 {
  fn default() -> Self {
    Self(0.0, 0.0)
  }
}

impl std::ops::Add for Vector2 {
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output {
    Self(self.0 + rhs.0, self.1 + rhs.1)
  }
}

impl std::ops::Sub for Vector2 {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output {
    Self(self.0 - rhs.0, self.1 - rhs.1)
  }
}

impl std::ops::Mul<f32> for Vector2 {
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output {
    Self(self.0 * rhs, self.1 * rhs)
  }
}

/// A physically simulated body, with optional data attached
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Body<T> {
  pub current_position: Vector2,
  pub last_position: Vector2,
  pub acceleration: Vector2,
  pub radius: f32,
  /// Arbitrary data such as color, text, etc
  pub payload: T,
}

impl<T: Default> Default for Body<T> {
  fn default() -> Self {
    Self {
      current_position: Default::default(),
      last_position: Default::default(),
      acceleration: Default::default(),
      radius: Default::default(),
      payload: Default::default(),
    }
  }
}

/// Perform verlet integration on a body, simulating the
/// effects of velocity and acceleration.
/// * `to_integrate` - Body to be simulated
/// * `gravity` - Acceleration which is applied to the body
/// * `dt2` - Time since the last integration squared
pub fn integrate<T>(
  to_integrate: Body<T>,
  gravity: Vector2,
  dt2: f32,
) -> Body<T> {
  Body {
    current_position: to_integrate.current_position
      + (to_integrate.current_position - to_integrate.last_position)
      + (to_integrate.acceleration * dt2),
    acceleration: gravity,
    ..to_integrate
  }
}

/// Simulate collisions between a set of bodies. As a side
/// effect, unstable sort the bodies by their X coordinate
/// (ascending).
pub fn collide_bodies<T>(mut bodies: Vec<Body<T>>) -> Vec<Body<T>> {
  const RESPONSE_MODIFIER: f32 = 0.4;
  bodies.sort_unstable_by(|a, b| {
    a.current_position.0.total_cmp(&b.current_position.0)
  });

  for i in 0..bodies.len() {
    for j in i..bodies.len() {
      let position1 = bodies[i].current_position;
      let position2 = bodies[j].current_position;
      let min_distance = bodies[i].radius + bodies[j].radius;
      // Three early outs to speed up the algorithm
      if position1.0 < position2.0 - min_distance {
        break; // No further collisions possible
      }
      let dy = position1.1 - position2.1;
      if dy.abs() >= min_distance {
        continue; // Skip obvious non-collisions
      }
      let dx = position1.0 - position2.0;
      let distance_squared = dx.powi(2) + dy.powi(2);
      if distance_squared >= min_distance.powi(2) || distance_squared == 0.0 {
        continue; // Final check
      }
      let distance = distance_squared.sqrt();
      let normal = Vector2(dx / distance, dy / distance);
      let mass_ratio_1 = bodies[i].radius / min_distance;
      let mass_ratio_2 = bodies[j].radius / min_distance;
      let delta = 0.5 * RESPONSE_MODIFIER * (distance - min_distance);
      bodies[i].current_position = bodies[i].current_position
        - Vector2(
          normal.0 * mass_ratio_2 * delta,
          normal.1 * mass_ratio_2 * delta,
        );
      bodies[j].current_position = bodies[j].current_position
        + Vector2(
          normal.0 * mass_ratio_1 * delta,
          normal.1 * mass_ratio_1 * delta,
        );
    }
  }
  bodies
}

/// Constrain a body to the area of a circle
pub fn constraint_circle<T>(
  mut body: Body<T>,
  center: Vector2,
  radius: f32,
) -> Body<T> {
  let diff = center - body.current_position;
  let distance = (diff.0.powi(2) + diff.1.powi(2)).sqrt();
  if distance > radius - body.radius {
    let normal = diff * (1.0 / distance);
    body.current_position = center - normal * (radius - body.radius);
  }
  body
}

/// Constrain a body to the area of a rectangle
pub fn constraint_rectangle<T>(
  circle: Body<T>,
  top_left: Vector2,
  bottom_right: Vector2,
) -> Body<T> {
  Body {
    current_position: Vector2(
      circle.current_position.0.clamp(top_left.0, bottom_right.0),
      circle.current_position.1.clamp(top_left.1, bottom_right.1),
    ),
    ..circle
  }
}

/// Physically simulate a set of bodies
///
/// # Arguments
/// * `bodies` - Bodies to be simulated
///
/// * `steps` - Number of simulations to perform. Higher =
///   slower, but more accurate
///
/// * `constraint` - Function applied to each body before
///   each simulation step. Can be used to constrain bodies
///   to a region
///
/// * `gravity` - Constant acceleration force to applied to
///   all bodies
///
/// * `time` - Duration of the simulation
pub fn simulate<T>(
  bodies: Vec<Body<T>>,
  steps: usize,
  constraint: fn(Body<T>) -> Body<T>,
  gravity: Vector2,
  time: std::time::Duration,
) -> Vec<Body<T>> {
  let dt2 = time.as_secs_f32().powi(2) / (steps as f32);
  (0..steps).into_iter().fold(bodies, |circles, _| {
    collide_bodies(circles)
      .into_iter()
      .map(|c| integrate(constraint(c), gravity, dt2))
      .collect()
  })
}
