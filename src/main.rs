use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops;
use std::{path::Path};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_DIMS: usize = 5;
const GRID_SIZE: usize = 128;
const SCALE: f64 = 32.0;
const GRID_LEN: usize = ((GRID_SIZE as f64 / SCALE) as usize + 3) * 2;

type Veci = [i64; MAX_DIMS];
type Vecf = [f64; MAX_DIMS];

static mut GRID_CACHE: [[[[[Vecf; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN] = [[[[[[0.0; MAX_DIMS]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN];
// static mut grad_cache: [[[[[Vecf; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN] = [[[[[[0.0; MAX_DIMS]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN]; GRID_LEN];

fn get_grid_cache(pos: &Veci) -> Vecf {
	unsafe {
		return GRID_CACHE[(pos[0] + GRID_LEN as i64 / 2) as usize]
		[(pos[1] + GRID_LEN as i64 / 2) as usize]
		[(pos[2] + GRID_LEN as i64 / 2) as usize]
		[(pos[3] + GRID_LEN as i64 / 2) as usize]
		[(pos[4] + GRID_LEN as i64 / 2) as usize];
	}
}

fn vecf<const T: usize>(v: [f64; T]) -> Vecf {
	let mut p: Vecf = [0.0; MAX_DIMS];
	for i in 0..T {
		p[i] = v[i];
	}
	return p;
}

fn veci<const T: usize>(v: [i64; T]) -> Veci {
	let mut p: Veci = [0; MAX_DIMS];
	for i in 0..T {
		p[i] = v[i];
	}
	return p;
}

fn veci2f(v: &Veci) -> Vecf {
	return v.map(|a| a as f64);
}

fn vecf2i(v: &Vecf) -> Veci {
	return v.map(|a| a as i64);
}

fn nearest_axis(v: &Vecf) -> Vecf {
	let mut index = 0;
	let mut max = v[0].abs();
	let mut axis = [0.0; MAX_DIMS];
	for i in 1..3 {
		if v[i].abs() > max {
			index = i;
			max = v[i].abs();
		}
	}

	axis[index] = max.signum();

	return axis;
}

fn sub(a: &Vecf, b: &Vecf, len: usize) -> Vecf {
	//maybe implement SIMD? maybe cheating
	let mut res = [0.0; MAX_DIMS];
	for i in 0..len {
		res[i] = a[i] - b[i];
	}
	return res;
}

fn add(a: &Vecf, b: &Vecf, len: usize) -> Vecf {
	//maybe implement SIMD? maybe cheating
	let mut res = [0.0; MAX_DIMS];
	for i in 0..len {
		res[i] = a[i] + b[i];
	}
	return res;
}

fn dot(a: &Vecf, b: &Vecf, len: usize) -> f64 {
	//maybe implement SIMD? maybe cheating
	let mut res = 0.0;
	for i in 0..len {
		res += a[i] * b[i];
	}
	return res;
}

fn mul(a: &Vecf, b: f64, len: usize) -> Vecf {
	//maybe implement SIMD? maybe cheating
	let mut res = [0.0; MAX_DIMS];
	for i in 0..len {
		res[i] = a[i] * b;
	}
	return res;
}

fn div(a: &Vecf, b: f64, len: usize) -> Vecf {
	//maybe implement SIMD? maybe cheating
	let mut res = [0.0; MAX_DIMS];
	for i in 0..len {
		res[i] = a[i] / b;
	}
	return res;
}

fn normalize(a: &Vecf) -> Vecf {
	let len = a.map(|x| x * x).iter().sum::<f64>().sqrt();
	//maybe implement SIMD? maybe cheating
	return std::array::from_fn(|i| a[i] / len);
}

fn smoothstep(x: f64) -> f64 {
	return 3.0 * x * x - 2.0 * x * x * x;
}

fn d_smoothstep(x: f64) -> f64 {
	return 6.0 * x - 6.0 * x * x;
}

fn int_hash(i: i64) -> i64 {
	let mut x = ((i >> 32) ^ i).wrapping_mul(-4776276827692571787);
    x = ((x >> 32) ^ x).wrapping_mul(-4776276827692571787);
    x = (x >> 32)  ^ x;
	return x;
}

fn int_vec_hash(vec: &Veci, start: i64, dims: usize) -> i64 {
	let mut hash = start;
	for i in 0..dims {
		hash = int_hash(hash.wrapping_mul(-4776276827692571787) + vec[i]);
	}
	return hash;
}

fn get_gridvalue_n(pos: &Veci, dims: usize) -> Vecf {
	let mut bm: Vecf = [0.0; MAX_DIMS];

	for i in 0..dims {
		let t = (int_vec_hash(&pos, (i * dims) as i64, dims) % 256) as f64 / 256.0 * 2.0 * PI;
		let r = (((int_vec_hash(&pos, (i * dims) as i64 + 1, dims) % 256 + 1) as f64 / 256.0).ln() * -2.0).sqrt();
		bm[i] = r * t.cos();
		// bm[i] = (int_vec_hash(pos, i as i64 * 11 as i64 + 1235, dims) % 256) as f64 / 255.0 * 2.0 - 1.0;
	}

	return bm;
}

// fn get_perlin(x: f64, y: f64) -> f64 {
//     let ffv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32) * PI * 2.0);
//     let fcv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32 + 1) * PI * 2.0);
//     let cfv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32) * PI * 2.0);
//     let ccv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32 + 1) * PI * 2.0);

//     let h1 = x - x.floor();
//     let v1 = y - y.floor();

//     let ff = Vec2 {x: h1, y: v1} ^ ffv;
//     let fc = Vec2 {x: h1, y: v1 - 1.0} ^ fcv;
//     let cf = Vec2 {x: h1 - 1.0, y: v1} ^ cfv;
//     let cc = Vec2 {x: h1 - 1.0, y: v1 - 1.0} ^ ccv;

//     let h = smoothstep(h1);
//     let v = smoothstep(v1);

//     let top = ff + (cf - ff) * h;
//     let bot = fc + (cc - fc) * h;
//     let total = top + (bot - top) * v;

//     return total * 0.5 + 0.5;
// }

fn get_perlin_n_inner(pos: &Vecf, origin: &Vecf, f_dims: usize, o_dims: usize) -> f64 {
	if f_dims == 0 {
		unsafe {
			let ipos = &vecf2i(pos);
			let v = get_grid_cache(ipos);
			let cv = sub(origin, pos, o_dims);
			let val = dot(&cv, &v, o_dims);
			return val;
		}
	} else {
		let mut fpos = *pos;
		let mut cpos = *pos;
		let dim = f_dims - 1;
		fpos[dim] = pos[dim].floor();
		cpos[dim] = pos[dim].ceil();

		let f = get_perlin_n_inner(&fpos, origin, dim, o_dims);
		let c = get_perlin_n_inner(&cpos, origin, dim, o_dims);

		let interp = smoothstep(pos[dim] - pos[dim].floor());
		let total = f + (c - f) * interp;

		return total;
	}
}

fn get_perlin_n(pos: &Vecf, dims: usize) -> f64 {
	let val = get_perlin_n_inner(pos, pos, dims, dims);

	return val;
}

fn get_perlin_n_grad_inner(pos: &Vecf, origin: &Vecf, f_dims: usize, o_dims: usize) -> (Vecf, f64) {
	if f_dims == 0 {
		unsafe {
			let ipos = &vecf2i(pos);
			let v = get_grid_cache(ipos);
			// let v = get_gridvalue_n(&vecf2i(pos), o_dims);
			let cv = sub(origin, pos, o_dims);
			let val = dot(&cv, &v, o_dims);
			return (v, val);
		}
	} else {
		let mut fpos = *pos;
		let mut cpos = *pos;
		fpos[f_dims - 1] = pos[f_dims - 1].floor();
		cpos[f_dims - 1] = pos[f_dims - 1].ceil();

		let (fv, f) = get_perlin_n_grad_inner(&fpos, origin, f_dims - 1, o_dims);
		let (cv, c) = get_perlin_n_grad_inner(&cpos, origin, f_dims - 1, o_dims);

		let interp = smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());
		let d_interp = d_smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());

		let val = f + (c - f) * interp;

		let mut total = add(&mul(&sub(&cv, &fv, o_dims), interp, o_dims), &fv, o_dims);
		total[f_dims - 1] += (c - f) * d_interp;

		return (total, val);
	}
}

fn get_perlin_grad_n(pos: &Vecf, dims: usize) -> (Vecf, f64) {
	let (vec, val) = get_perlin_n_grad_inner(pos, pos, dims, dims);

	return (normalize(&vec), val);
}

fn render_2d(pixels: Vec<Vec<f64>>) -> Vec<u8> {
	return pixels.iter().flatten().collect::<Vec<&f64>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

fn grey_to_rgb(pixels: Vec<u8>) -> Vec<u8> {
	return pixels.iter().map(|v| vec![*v, *v, *v]).collect::<Vec<Vec<u8>>>().iter().flatten().collect::<Vec<&u8>>().iter().map(|v| **v).collect::<Vec<u8>>();
}

fn vecf_to_rgb(pixels: Vec<Vecf>) -> Vec<u8> {
	return pixels.iter().map(|v| vec![(v[0] * 255.0) as u8, (v[1] * 255.0) as u8, (v[2] * 255.0) as u8]).collect::<Vec<Vec<u8>>>().iter().flatten().collect::<Vec<&u8>>().iter().map(|v| **v).collect::<Vec<u8>>();
}

fn render_3d(grid_size: usize, scale: f64, threshold: f64, w1: f64, w2: f64) -> Vec<u8> {
	let mut pixels = vec![vec![vec![0.0, 0.0, 0.0]; grid_size]; grid_size];
	let mut depth_buffer = vec![vec![false; grid_size]; grid_size];

	for i in (0..grid_size).rev() {
		println!("Rendering slice {i}/{grid_size}");
		for j in (0..grid_size).rev() {
			for k in 0..grid_size {
				let gy = grid_size - 1 - (j * 7 / 10 + k * 3 / 10);
				let gx = i * 7 / 10 + k * 3 / 10;
				if !depth_buffer[gy][gx] {
					let pi = &veci([i as i64, j as i64, k as i64]);
					let p = &vecf([i as f64 / scale, j as f64 / scale, k as f64 / scale, w1, w2]);
					let (norm, val) = get_perlin_grad_n(p, 5);
					if val < threshold {
						let n = 
						if val < threshold - 0.1 {
							let mut nn = pi.map(|a| if (a == 0) {-1.0} else if (a == (grid_size as i64 - 1)) {1.0} else {0.0});
							nn[2] = nn[2];
							nn
						} else {
							norm
						};
	
						// pixels[gx][gy][2] = (-t[2]).max(0.0);
						depth_buffer[gy][gx] = true;
	
						// pixels[gy][gx][0] = n[0] * 0.5 + 0.5;
						// pixels[gy][gx][1] = n[1] * 0.5 + 0.5;
						// pixels[gy][gx][2] = n[2] * 0.5 + 0.5;
						pixels[gy][gx][0] = n[0];
						pixels[gy][gx][1] = n[1];
						pixels[gy][gx][2] = -n[2];
					}
				}
			}
		}
	}

	return pixels.iter().flatten().collect::<Vec<&Vec<f64>>>().iter().map(|v| *v).flatten().collect::<Vec<_>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

fn render_3d_good(grid_size: usize, scale: f64, threshold: f64, w1: f64, w2: f64) -> Vec<u8> {
	let mut pixels = vec![vec![vec![0.0, 0.0, 0.0]; grid_size]; grid_size];

	let mut total = 0;

	for x in 0..grid_size {
		// println!("Rendering slice {x}/{grid_size}");
		for y in 0..grid_size {
			let fx = (x as f64 - grid_size as f64 / 2.0) / scale;
			let fy = (y as f64 - grid_size as f64 / 2.0) / scale;
			let fz = (4.0 - fx * fx - fy * fy).sqrt();
			let mut z = -fz;
			// if fx * fx + fy * fy > 4.0 {
			// 	continue;
			// }
			while z < fz as f64 {
				total += 1;
				let p = &vecf([fx, fy, z, w1, w2]);
				let len_s = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
				if len_s > 4.0 {
					z += 1.0 / scale;
					continue;
				}
				let (norm, val) = get_perlin_grad_n(p, 5);
				if val < threshold {
					let n = 
					if val < threshold - 0.1 {
						// nearest_axis(p)
						div(p, len_s.sqrt(), 5)
					} else {
						norm
					};
		
					// pixels[gy][gx][0] = n[0] * 0.5 + 0.5;
					// pixels[gy][gx][1] = n[1] * 0.5 + 0.5;
					// pixels[gy][gx][2] = n[2] * 0.5 + 0.5;
					pixels[grid_size - y - 1][x][0] = n[0];
					pixels[grid_size - y - 1][x][1] = n[1];
					pixels[grid_size - y - 1][x][2] = n[2];
				}
				z += (val * 1.4 / scale).abs().max(1.0 / scale);
				// z += 1.0;
			}
		}
	}

	return pixels.iter().flatten().collect::<Vec<&Vec<f64>>>().iter().map(|v| *v).flatten().collect::<Vec<_>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

//main image, gradient map
fn render_2d_in_4d(grid_size: usize, scale: f64, z: f64, w: f64) -> Vec<u8> {
    let pixels: Vec<Vec<f64>> = (0..grid_size).map(|y| (0..grid_size).map(|x| (get_perlin_n(&vecf([x as f64 / scale, y as f64 / scale, z, w]), 4) * 0.75 + 0.5)).collect::<Vec<f64>>()).collect::<Vec<_>>();
    let buffer = grey_to_rgb(render_2d(pixels));

    return buffer;
}

fn main() {
	unsafe {
		for x in 0..GRID_LEN {
			for y in 0..GRID_LEN {
				for z in 0..GRID_LEN {
					for w1 in 0..GRID_LEN {
						for w2 in 0..GRID_LEN {
							GRID_CACHE[x][y][z][w1][w2] = get_gridvalue_n(&[x as i64, y as i64, z as i64, w1 as i64, w2 as i64], MAX_DIMS);
						}
					}
				}
			}
		}
	}

	// for i in 0..101 {
	// 	let buffer = render_3d(grid_size, scale, (i - 50) as f64 * 0.015);

	// 	let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, grid_size as u32, grid_size as u32, image::ExtendedColorType::Rgb8);
	// 	if error.is_err() {
	// 		println!("Fail :(");
	// 	} else {
	// 		println!("Finished image{i}.png");
	// 	}
	// }

	// for i in 0..100 {
	// 	let a = i as f64 / 100.0 * PI * 2.0;
	// 	let buffer = render_2d_in_4d(GRID_SIZE, SCALE, a.cos() * 2.0, a.sin() * 2.0);

	// 	let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, GRID_SIZE as u32, GRID_SIZE as u32, image::ExtendedColorType::Rgb8);
	// 	if error.is_err() {
	// 		println!("Fail :(");
	// 	} else {
	// 		println!("Finished image{i}.png");
	// 	}
	// }

	// let start = SystemTime::now().duration_since(UNIX_EPOCH).expect("time backwards?").as_millis();

	// let mut accum = 0.0;
	// for i in 0..10000 {
	// 	accum += get_perlin_n(&([i as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5);
	// 	// accum += get_gridvalue_n(&([i, 0, 0, 0, 0, 0, 0, 0]), 5)[0];
	// }

	// let end = SystemTime::now().duration_since(UNIX_EPOCH).expect("time backwards?").as_millis();

	// let len = end - start;

	// println!("Total time: {}ms, Time per iteration: {}us/{}ns, accum: {}", len, len / 10, len * 100, accum);

	for i in 0..100 {
		let a = i as f64 / 100.0 * PI * 2.0;
		let buffer = render_3d_good(GRID_SIZE, SCALE, 0.0, a.cos() * 1.5 + 2.0, a.sin() * 1.5 + 2.0);

		let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, GRID_SIZE as u32, GRID_SIZE as u32, image::ExtendedColorType::Rgb8);
		if error.is_err() {
			println!("Fail :(");
		} else {
			println!("Finished image{i}.png");
		}
	}

}
