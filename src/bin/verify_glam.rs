use glam::{Mat4, Vec4};

fn main() {
    let fov = 45.0_f32.to_radians();
    let aspect = 1.0;
    let near = 0.1;
    let far = 100.0;

    let proj = Mat4::perspective_rh(fov, aspect, near, far);

    let p_near = Vec4::new(0.0, 0.0, -near, 1.0);
    let p_far = Vec4::new(0.0, 0.0, -far, 1.0);

    let ndc_near = proj * p_near;
    let ndc_far = proj * p_far;

    println!("Near Z (NDC): {}", ndc_near.z / ndc_near.w);
    println!("Far Z (NDC): {}", ndc_far.z / ndc_far.w);
}
