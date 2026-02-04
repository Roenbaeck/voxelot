// WTS-RT: Symplectic Ray Relaxation Compute Shader
// Solves: Phi_{t+1} = Phi_t + alpha * (Grad^2 Phi - Phi/Beta) + gamma * (Seed - Phi)
// Inspired by: Aaron M. Schutza, "Symplectic Ray Relaxation: An O(1) Global Illumination Method
// via Geometric Stress Minimization on Tensor Architecture" (Jan 7, 2026).

struct WtsParams {
    alpha: f32,          // Relaxation rate (diffusion speed)
    gamma: f32,          // Injection rate (seed influence)
    beta_diffuse: f32,   // Stiffness for diffuse surfaces (low)
    beta_air: f32,       // Stiffness for air (high)
    max_occupancy: f32,
    decay: f32,          // Per-step decay to bleed off stale light
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: WtsParams;
@group(0) @binding(1) var seed_texture: texture_3d<f32>; // Raw GI Probe Color from CPU
@group(0) @binding(2) var phi_in: texture_3d<f32>;       // Relaxed field from previous frame
@group(0) @binding(3) var phi_out: texture_storage_3d<rgba16float, write>; // New relaxed field
@group(0) @binding(4) var injection_texture: texture_3d<f32>; // Sunlight injection grid

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(phi_out);
    if (any(id >= dims)) {
        return;
    }

    let coord = vec3<i32>(id);
    var phi_t = textureLoad(phi_in, coord, 0);
    // Use select or simple comparison if isNan/isInf not available in this WGSL version
    if (!(all(phi_t.rgb >= vec3<f32>(-1e6)) && all(phi_t.rgb <= vec3<f32>(1e6)))) {
        phi_t = vec4<f32>(0.0, 0.0, 0.0, phi_t.a);
    }
    let seed = textureLoad(seed_texture, coord, 0);

    // 7-point stencil for 3D Laplacian (Grad^2 Phi)
    var laplacian = vec3<f32>(0.0);
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1)
    );

    var neighbor_count = 0.0;
    for (var i = 0u; i < 6u; i++) {
        let n_coord = coord + neighbors[i];
        if (all(n_coord >= vec3<i32>(0)) && all(n_coord < vec3<i32>(dims))) {
            let neighbor_phi = textureLoad(phi_in, n_coord, 0).rgb;
            if (all(neighbor_phi >= vec3<f32>(-1e6)) && all(neighbor_phi <= vec3<f32>(1e6))) {
                laplacian += neighbor_phi - phi_t.rgb;
                neighbor_count += 1.0;
            }
        }
    }
    
    // Stiffness Beta: High in air (diffuse far), Low in solids (localize)
    // Actually, according to the paper: Beta -> inf for mirror, Beta -> 0 for diffuse.
    // In our simplified model, we use occupancy as the "diffuse" factor.
    let occupancy = seed.a;
    let beta = mix(params.beta_air, params.beta_diffuse, clamp(occupancy / params.max_occupancy, 0.0, 1.0));
    
    // Relaxation Step
    // Screen term: -Phi/Beta. If beta is small (solid), this term is large, damping the potential.
    let screen_term = phi_t.rgb / (beta + 1e-4);
    let relaxation = params.alpha * (laplacian - screen_term);
    
    // Injection: Sunlight/Emissive flux from injection_texture
    let flux = textureLoad(injection_texture, coord, 0).rgb;
    
    // Update
    // alpha * flux treats injected light as a source term for the diffusion equation.
    var new_phi_rgb = phi_t.rgb + relaxation + params.alpha * flux;
    new_phi_rgb *= (1.0 - params.decay);
    
    // Clamp to prevent runaway/NaNs (limited for LDR/balanced HDR)
    new_phi_rgb = clamp(new_phi_rgb, vec3<f32>(0.0), vec3<f32>(8.0));
    
    // Preserve occupancy in alpha for SSR
    textureStore(phi_out, coord, vec4<f32>(new_phi_rgb, occupancy));
}
