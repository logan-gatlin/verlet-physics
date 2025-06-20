Physics simulation for circles using verlet integration.

# Usage
```rust
extern crate verlet;
let mut bodies = vec![];
for i in 0..10 {
  bodies.push(Body {
    current_position: Vector2(i as f32, 0.0),
    radius: 2.0,
    payload: (),
    ..Default::default()
  })
}
// Simulate the bodies 100 times over the course of 30 seconds
simulate(
  bodies,
  100,
  |c| constraint_circle(c, Vector2(0.0, 0.0), 10.0),
  Vector2(0.0, 5.0),
  std::time::Duration::from_secs(30),
);
```
