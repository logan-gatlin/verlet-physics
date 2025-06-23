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

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// An (x, y) coordinate
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "wasm", wasm_bindgen(inspectable))]
pub struct Vector2 {
  pub x: f32,
  pub y: f32,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl Vector2 {
  #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
  pub fn new(x: f32, y: f32) -> Vector2 {
    Vector2 { x, y }
  }
}

impl Default for Vector2 {
  fn default() -> Self {
    Self::new(0.0, 0.0)
  }
}

impl std::ops::Add for Vector2 {
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output {
    Self::new(self.x + rhs.x, self.y + rhs.y)
  }
}

impl std::ops::Sub for Vector2 {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output {
    Self::new(self.x - rhs.x, self.y - rhs.y)
  }
}

impl std::ops::Mul<f32> for Vector2 {
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output {
    Self::new(self.x * rhs, self.y * rhs)
  }
}

/// A physically simulated body, with optional data attached
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "wasm", wasm_bindgen(inspectable))]
pub struct Body {
  pub current_position: Vector2,
  pub last_position: Vector2,
  pub acceleration: Vector2,
  pub radius: f32,
  pub index: usize,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl Body {
  #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
  pub fn new(position: Vector2, radius: f32, index: usize) -> Self {
    Body {
      current_position: position,
      last_position: position,
      acceleration: Vector2::default(),
      radius,
      index,
    }
  }
}

impl Default for Body {
  fn default() -> Self {
    Self {
      current_position: Default::default(),
      last_position: Default::default(),
      acceleration: Default::default(),
      radius: Default::default(),
      index: Default::default(),
    }
  }
}

/// Perform verlet integration on a body, simulating the
/// effects of velocity and acceleration.
/// * `to_integrate` - Body to be simulated
/// * `gravity` - Acceleration which is applied to the body
/// * `dt2` - Time since the last integration squared
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn integrate(to_integrate: Body, gravity: Vector2, dt: f32) -> Body {
  Body {
    current_position: to_integrate.current_position
      + (to_integrate.current_position - to_integrate.last_position)
      + (to_integrate.acceleration * dt),
    last_position: to_integrate.current_position,
    acceleration: gravity,
    ..to_integrate
  }
}

/// Simulate collisions between a set of bodies. As a side
/// effect, unstable sort the bodies by their X coordinate
/// (ascending).
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn collide_bodies(mut bodies: Vec<Body>) -> Vec<Body> {
  const RESPONSE_MODIFIER: f32 = 0.4;
  bodies.sort_unstable_by(|a, b| {
    let a_left = a.current_position.x - a.radius;
    let b_left = b.current_position.x - b.radius;
    a_left.total_cmp(&b_left)
  });

  for i in 0..bodies.len() {
    let mut last = bodies.len();
    for j in i..bodies.len() {
      let position1 = bodies[i].current_position;
      let position2 = bodies[j].current_position;
      let min_distance = bodies[i].radius + bodies[j].radius;
      if position2.x > position1.x + min_distance {
        last = j;
        break;
      }
    }

    for j in i..last {
      let position1 = bodies[i].current_position;
      let position2 = bodies[j].current_position;
      let min_distance = bodies[i].radius + bodies[j].radius;
      let difference = position1 - position2;
      let distance_squared = difference.x.powi(2) + difference.y.powi(2);
      if distance_squared >= min_distance.powi(2) || distance_squared == 0.0 {
        continue; // Not colliding
      }
      let distance = distance_squared.sqrt();
      let normal = difference * (1.0 / distance);
      let mass_ratio_1 = bodies[i].radius / min_distance;
      let mass_ratio_2 = bodies[j].radius / min_distance;
      let delta = 0.5 * RESPONSE_MODIFIER * (distance - min_distance);
      bodies[i].current_position =
        bodies[i].current_position - normal * mass_ratio_2 * delta;
      bodies[j].current_position =
        bodies[j].current_position + normal * mass_ratio_1 * delta;
    }
  }
  bodies
}

/// Constrain a body to the area of a circle
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn constraint_circle(mut body: Body, center: Vector2, radius: f32) -> Body {
  let diff = center - body.current_position;
  let distance = (diff.x.powi(2) + diff.y.powi(2)).sqrt();
  if distance > radius - body.radius {
    let normal = diff * (1.0 / distance);
    body.current_position = center - normal * (radius - body.radius);
  }
  body
}

/// Constrain a body to the area of a rectangle
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn constraint_rectangle(
  circle: Body,
  top_left: Vector2,
  bottom_right: Vector2,
) -> Body {
  let constrained = Vector2::new(
    circle
      .current_position
      .x
      .clamp(top_left.x + circle.radius, bottom_right.x - circle.radius),
    circle
      .current_position
      .y
      .clamp(top_left.y + circle.radius, bottom_right.y - circle.radius),
  );
  if constrained != circle.current_position {
    Body {
      current_position: constrained,
      last_position: circle.current_position,
      ..circle
    }
  } else {
    circle
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
#[cfg(feature = "wasm")]
pub fn simulate(
  bodies: Vec<Body>,
  steps: usize,
  constraint: impl Fn(Body) -> Body,
  gravity: Vector2,
  time: std::time::Duration,
) -> Vec<Body> {
  let dt2 = (time.as_secs_f32() / (steps as f32)).powi(2);
  println!("{dt2}");
  (0..steps).into_iter().fold(bodies, |bodies, _| {
    collide_bodies(bodies)
      .into_iter()
      .map(|c| integrate(constraint(c), gravity, dt2))
      .collect()
  })
}

/// Same as `simulate` with a circle constraint for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn simulate_circle(
  bodies: Vec<Body>,
  steps: usize,
  time: f32,
  center: Vector2,
  radius: f32,
  gravity: Vector2,
) -> Vec<Body> {
  simulate(
    bodies,
    steps,
    &|c| constraint_circle(c, center, radius),
    gravity,
    std::time::Duration::from_secs_f32(time),
  )
}

/// Same as `simulate` with a rectangle constraint for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn simulate_rectangle(
  bodies: Vec<Body>,
  steps: usize,
  time: f32,
  top_left: Vector2,
  bottom_right: Vector2,
  gravity: Vector2,
) -> Vec<Body> {
  simulate(
    bodies,
    steps,
    &|c| constraint_rectangle(c, top_left, bottom_right),
    gravity,
    std::time::Duration::from_secs_f32(time),
  )
}
